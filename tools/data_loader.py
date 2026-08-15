"""Datasets and batches for translation and decoder-only language modeling."""

import json
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

import config
from tools.tokenizer_utils import chinese_tokenizer_load, english_tokenizer_load


def subsequent_mask(size):
    """Return an autoregressive mask that hides future decoder positions."""
    return torch.tril(torch.ones((1, size, size), dtype=torch.bool))


def normalize_text(text):
    """Normalize whitespace without changing the content or case of a document."""
    return " ".join(text.split())


def clean_decoder_only_texts(texts, min_characters=2, deduplicate=True):
    """Filter empty/short documents and optionally remove exact duplicates."""
    cleaned = []
    seen = set()
    statistics = {
        "raw_documents": len(texts),
        "kept_documents": 0,
        "dropped_empty": 0,
        "dropped_too_short": 0,
        "dropped_duplicates": 0,
        "characters": 0,
        "min_characters": 0,
        "max_characters": 0,
        "mean_characters": 0.0,
    }
    for text in texts:
        normalized = normalize_text(text)
        if not normalized:
            statistics["dropped_empty"] += 1
            continue
        if len(normalized) < min_characters:
            statistics["dropped_too_short"] += 1
            continue
        if deduplicate and normalized in seen:
            statistics["dropped_duplicates"] += 1
            continue
        seen.add(normalized)
        cleaned.append(normalized)

    lengths = [len(text) for text in cleaned]
    statistics["kept_documents"] = len(cleaned)
    if lengths:
        statistics["characters"] = sum(lengths)
        statistics["min_characters"] = min(lengths)
        statistics["max_characters"] = max(lengths)
        statistics["mean_characters"] = statistics["characters"] / len(lengths)
    return cleaned, statistics


class Batch:
    """A machine-translation batch, moved once to the selected device."""

    def __init__(self, src_text, trg_text, src, trg=None, pad=0, device=None):
        device = config.device if device is None else device
        self.src_text = src_text
        self.trg_text = trg_text
        self.src = src.to(device)
        self.src_mask = (self.src != pad).unsqueeze(-2)

        if trg is not None:
            target = trg.to(device)
            self.trg = target[:, :-1]
            self.trg_y = target[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        return tgt_mask & subsequent_mask(tgt.size(-1)).to(tgt.device)


class MTDataset(Dataset):
    """English-to-Chinese sentence-pair dataset from the original JSON format."""

    def __init__(self, data_path):
        self.out_en_sent, self.out_cn_sent = self.get_dataset(data_path, sort=True)
        self.sp_eng = english_tokenizer_load()
        self.sp_chn = chinese_tokenizer_load()
        self.PAD = self.sp_eng.pad_id()
        self.BOS = self.sp_eng.bos_id()
        self.EOS = self.sp_eng.eos_id()

    @staticmethod
    def get_dataset(data_path, sort=False):
        with Path(data_path).open("r", encoding="utf-8") as handle:
            dataset = json.load(handle)
        source, target = zip(*dataset) if dataset else ([], [])
        source, target = list(source), list(target)
        if sort:
            order = sorted(range(len(source)), key=lambda index: len(source[index]))
            source = [source[index] for index in order]
            target = [target[index] for index in order]
        return source, target

    def __getitem__(self, index):
        return self.out_en_sent[index], self.out_cn_sent[index]

    def __len__(self):
        return len(self.out_en_sent)

    def collate_fn(self, batch):
        src_text = [item[0] for item in batch]
        tgt_text = [item[1] for item in batch]
        src_tokens = [
            [self.BOS] + self.sp_eng.EncodeAsIds(sentence) + [self.EOS]
            for sentence in src_text
        ]
        tgt_tokens = [
            [self.BOS] + self.sp_chn.EncodeAsIds(sentence) + [self.EOS]
            for sentence in tgt_text
        ]
        src = pad_sequence(
            [torch.tensor(ids, dtype=torch.long) for ids in src_tokens],
            batch_first=True,
            padding_value=self.PAD,
        )
        tgt = pad_sequence(
            [torch.tensor(ids, dtype=torch.long) for ids in tgt_tokens],
            batch_first=True,
            padding_value=self.PAD,
        )
        return Batch(src_text, tgt_text, src, tgt, self.PAD)


class DecoderOnlyBatch:
    """A next-token-prediction batch for a causal language model."""

    def __init__(self, texts, tokens, pad=0, device=None):
        device = config.device if device is None else device
        self.texts = texts
        tokens = tokens.to(device)
        self.input_tokens = tokens[:, :-1]
        self.target_tokens = tokens[:, 1:]
        self.attention_mask = Batch.make_std_mask(self.input_tokens, pad)
        self.ntokens = (self.target_tokens != pad).sum()


class DecoderOnlyDataset(Dataset):
    """Text dataset for decoder-only next-token prediction.

    A UTF-8 text file contains one training document per non-empty line. A JSON
    dataset must be an array of strings or ``{"text": "..."}`` items.
    """

    def __init__(
        self,
        data_path,
        tokenizer_name="english",
        max_length=None,
        min_characters=None,
        deduplicate=None,
    ):
        if tokenizer_name not in {"english", "chinese"}:
            raise ValueError("tokenizer_name must be 'english' or 'chinese'")
        self.tokenizer = (
            english_tokenizer_load() if tokenizer_name == "english" else chinese_tokenizer_load()
        )
        self.max_length = max_length or config.decoder_only_max_sequence_length
        if self.max_length < 2:
            raise ValueError("decoder_only_max_sequence_length must be at least 2")
        self.pad = self.tokenizer.pad_id()
        self.bos = self.tokenizer.bos_id()
        self.eos = self.tokenizer.eos_id()
        raw_items = self._load_items(Path(data_path))
        self.items, self.statistics = clean_decoder_only_texts(
            raw_items,
            config.decoder_only_min_characters if min_characters is None else min_characters,
            config.decoder_only_deduplicate if deduplicate is None else deduplicate,
        )
        if not self.items:
            raise ValueError(f"Decoder-only dataset is empty: {data_path}")

    @classmethod
    def _load_items(cls, data_path):
        if data_path.suffix.lower() == ".json":
            with data_path.open("r", encoding="utf-8") as handle:
                raw_items = json.load(handle)
            if not isinstance(raw_items, list):
                raise ValueError("Decoder-only JSON data must be an array")
            return [cls._parse_item(item, index) for index, item in enumerate(raw_items)]
        with data_path.open("r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]

    @staticmethod
    def _parse_item(item, index):
        text = item if isinstance(item, str) else item.get("text") if isinstance(item, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Decoder-only sample {index} requires non-empty text")
        return text

    def __getitem__(self, index):
        return self.items[index]

    def __len__(self):
        return len(self.items)

    def collate_fn(self, batch):
        texts = list(batch)
        token_lists = []
        for text in texts:
            token_ids = self.tokenizer.EncodeAsIds(text)[: max(self.max_length - 2, 0)]
            token_lists.append([self.bos] + token_ids + [self.eos])
        tokens = pad_sequence(
            [torch.tensor(ids, dtype=torch.long) for ids in token_lists],
            batch_first=True,
            padding_value=self.pad,
        )
        return DecoderOnlyBatch(texts, tokens, self.pad)
