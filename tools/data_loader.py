"""Datasets and batches for translation and encoder-only tasks."""

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


class EncoderOnlyBatch:
    """A labeled text batch for encoder-only training and evaluation."""

    def __init__(self, texts, tokens, labels, pad=0, device=None):
        device = config.device if device is None else device
        self.texts = texts
        self.src = tokens.to(device)
        self.src_mask = (self.src != pad).unsqueeze(-2)
        self.labels = labels.to(device)


class EncoderOnlyDataset(Dataset):
    """JSON text classification dataset for the encoder-only Transformer.

    Accepted JSON items are ``{"text": "a sentence", "label": 0}`` or the
    shorter ``["a sentence", 0]`` form. Labels must be integer class IDs.
    """

    def __init__(self, data_path, tokenizer_name="english", max_length=None):
        if tokenizer_name not in {"english", "chinese"}:
            raise ValueError("tokenizer_name must be 'english' or 'chinese'")
        self.tokenizer = (
            english_tokenizer_load() if tokenizer_name == "english" else chinese_tokenizer_load()
        )
        self.max_length = max_length or config.max_source_len
        self.pad = self.tokenizer.pad_id()
        self.bos = self.tokenizer.bos_id()
        self.eos = self.tokenizer.eos_id()
        with Path(data_path).open("r", encoding="utf-8") as handle:
            raw_items = json.load(handle)
        self.items = [self._parse_item(item, index) for index, item in enumerate(raw_items)]

    @staticmethod
    def _parse_item(item, index):
        if isinstance(item, dict):
            text, label = item.get("text"), item.get("label")
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            text, label = item
        else:
            raise ValueError(f"Invalid encoder-only sample at index {index}")
        if not isinstance(text, str) or not isinstance(label, int):
            raise ValueError(f"Sample {index} requires string 'text' and integer 'label'")
        return text, label

    def __getitem__(self, index):
        return self.items[index]

    def __len__(self):
        return len(self.items)

    def collate_fn(self, batch):
        texts = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        token_lists = []
        for text in texts:
            token_ids = self.tokenizer.EncodeAsIds(text)[: max(self.max_length - 2, 0)]
            token_lists.append([self.bos] + token_ids + [self.eos])
        tokens = pad_sequence(
            [torch.tensor(ids, dtype=torch.long) for ids in token_lists],
            batch_first=True,
            padding_value=self.pad,
        )
        return EncoderOnlyBatch(texts, tokens, labels, self.pad)
