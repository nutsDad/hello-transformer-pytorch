"""Config-driven training and evaluation entry point."""

import argparse
import logging
from pathlib import Path

import sacrebleu
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from beam_decoder import beam_search
from model.checkpoint_utils import load_checkpoint, save_checkpoint
from model.tf_model import make_encoder_only_model, make_model
from model.train_utils import ClassificationLossCompute, TokenLossCompute, get_std_opt
from tools.data_loader import EncoderOnlyDataset, MTDataset
from tools.tokenizer_utils import chinese_tokenizer_load


logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
)


def make_dataloader(dataset, shuffle):
    kwargs = {
        "batch_size": config.batch_size,
        "shuffle": shuffle,
        "collate_fn": dataset.collate_fn,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
    }
    if config.num_workers > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(dataset, **kwargs)


def maybe_parallel(model):
    if config.use_data_parallel:
        logging.info("Using DataParallel on CUDA devices: %s", config.device_ids)
        return torch.nn.DataParallel(model, device_ids=config.device_ids)
    return model


def build_model():
    if config.model_architecture == "encoder_only":
        return make_encoder_only_model(
            config.encoder_only_vocab_size,
            config.encoder_only_num_labels,
            config.n_layers,
            config.d_model,
            config.d_ff,
            config.n_heads,
            config.dropout,
            config.encoder_only_pooling,
        )
    return make_model(
        config.src_vocab_size,
        config.tgt_vocab_size,
        config.n_layers,
        config.d_model,
        config.d_ff,
        config.n_heads,
        config.dropout,
    )


def create_weights_folder():
    train_dir = Path(config.run_dir) / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    index = 0
    while True:
        name = "exp" if index == 0 else f"exp{index}"
        folder = train_dir / name
        if not folder.exists():
            weights = folder / "weights"
            weights.mkdir(parents=True)
            return weights
        index += 1


def run_translation_epoch(data, model, loss_compute):
    total_loss = 0.0
    total_tokens = 0
    for batch in tqdm(data, leave=False):
        output = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        total_loss += loss_compute(output, batch.trg_y, batch.ntokens)
        total_tokens += batch.ntokens.item()
    return total_loss / max(total_tokens, 1)


def translation_bleu(data, model):
    tokenizer = chinese_tokenizer_load()
    targets, predictions = [], []
    with torch.no_grad():
        for batch in tqdm(data, leave=False):
            hypotheses, _ = beam_search(
                model,
                batch.src,
                batch.src_mask,
                config.max_len,
                config.padding_idx,
                config.bos_idx,
                config.eos_idx,
                config.beam_size,
                config.device,
            )
            predictions.extend(tokenizer.decode_ids(item[0]) for item in hypotheses)
            targets.extend(batch.trg_text)
    return float(sacrebleu.corpus_bleu(predictions, [targets], tokenize="zh").score)


def train_translation(model):
    train_data = make_dataloader(MTDataset(config.train_data_path), shuffle=True)
    dev_data = make_dataloader(MTDataset(config.dev_data_path), shuffle=False)
    parallel_model = maybe_parallel(model)
    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=config.padding_idx,
        reduction="sum",
    )
    optimizer = get_std_opt(model)
    weights_folder = create_weights_folder()
    best_bleu = float("-inf")

    for epoch in range(1, config.epoch_num + 1):
        model.train()
        train_loss = run_translation_epoch(
            train_data,
            parallel_model,
            TokenLossCompute(model.generator, criterion, optimizer),
        )
        model.eval()
        with torch.no_grad():
            dev_loss = run_translation_epoch(
                dev_data,
                parallel_model,
                TokenLossCompute(model.generator, criterion),
            )
        bleu = translation_bleu(dev_data, model)
        logging.info(
            "epoch=%d train_loss=%.4f dev_loss=%.4f bleu=%.2f",
            epoch,
            train_loss,
            dev_loss,
            bleu,
        )
        if bleu > best_bleu:
            best_bleu = bleu
            save_checkpoint(
                weights_folder / "best_bleu.pth",
                model,
                "encoder_decoder",
                {"bleu": bleu, "epoch": epoch},
            )
        save_checkpoint(
            weights_folder / "last.pth",
            model,
            "encoder_decoder",
            {"bleu": bleu, "epoch": epoch},
        )


def evaluate_translation(model):
    load_checkpoint(
        config.inference_model_path,
        model,
        config.device,
        "encoder_decoder",
    )
    model.eval()
    test_data = make_dataloader(MTDataset(config.test_data_path), shuffle=False)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=config.padding_idx, reduction="sum")
    with torch.no_grad():
        loss = run_translation_epoch(test_data, model, TokenLossCompute(model.generator, criterion))
    bleu = translation_bleu(test_data, model)
    logging.info("test_loss=%.4f bleu=%.2f", loss, bleu)


def run_encoder_only_epoch(data, model, loss_compute):
    total_loss = 0.0
    total_examples = 0
    correct = 0
    for batch in tqdm(data, leave=False):
        logits = model(batch.src, batch.src_mask)
        total_loss += loss_compute(logits, batch.labels) * batch.labels.size(0)
        total_examples += batch.labels.size(0)
        correct += (logits.argmax(dim=-1) == batch.labels).sum().item()
    return total_loss / max(total_examples, 1), correct / max(total_examples, 1)


def make_encoder_only_data(path, shuffle):
    dataset = EncoderOnlyDataset(
        path,
        tokenizer_name=config.encoder_only_tokenizer,
        max_length=config.max_source_len,
    )
    labels = [label for _, label in dataset.items]
    if labels and (min(labels) < 0 or max(labels) >= config.encoder_only_num_labels):
        raise ValueError("Encoder-only labels must be in [0, encoder_only_num_labels).")
    return make_dataloader(dataset, shuffle=shuffle)


def train_encoder_only(model):
    train_data = make_encoder_only_data(config.encoder_only_train_data_path, shuffle=True)
    dev_data = make_encoder_only_data(config.encoder_only_dev_data_path, shuffle=False)
    parallel_model = maybe_parallel(model)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = get_std_opt(model)
    weights_folder = create_weights_folder()
    best_accuracy = float("-inf")

    for epoch in range(1, config.epoch_num + 1):
        model.train()
        train_loss, train_accuracy = run_encoder_only_epoch(
            train_data,
            parallel_model,
            ClassificationLossCompute(criterion, optimizer),
        )
        model.eval()
        with torch.no_grad():
            dev_loss, dev_accuracy = run_encoder_only_epoch(
                dev_data,
                parallel_model,
                ClassificationLossCompute(criterion),
            )
        logging.info(
            "epoch=%d train_loss=%.4f train_acc=%.4f dev_loss=%.4f dev_acc=%.4f",
            epoch,
            train_loss,
            train_accuracy,
            dev_loss,
            dev_accuracy,
        )
        if dev_accuracy > best_accuracy:
            best_accuracy = dev_accuracy
            save_checkpoint(
                weights_folder / "best_accuracy.pth",
                model,
                "encoder_only",
                {"accuracy": dev_accuracy, "epoch": epoch},
            )
        save_checkpoint(
            weights_folder / "last.pth",
            model,
            "encoder_only",
            {"accuracy": dev_accuracy, "epoch": epoch},
        )


def evaluate_encoder_only(model):
    load_checkpoint(config.inference_model_path, model, config.device, "encoder_only")
    model.eval()
    test_data = make_encoder_only_data(config.encoder_only_test_data_path, shuffle=False)
    with torch.no_grad():
        loss, accuracy = run_encoder_only_epoch(
            test_data,
            model,
            ClassificationLossCompute(torch.nn.CrossEntropyLoss()),
        )
    logging.info("test_loss=%.4f test_acc=%.4f", loss, accuracy)


def run(action=None):
    action = action or config.run_action
    if action not in {"train", "evaluate"}:
        raise ValueError("action must be 'train' or 'evaluate'")
    logging.info(
        "profile=%s device=%s architecture=%s action=%s",
        config.runtime_profile,
        config.device,
        config.model_architecture,
        action,
    )
    model = build_model()
    if config.model_architecture == "encoder_only":
        return train_encoder_only(model) if action == "train" else evaluate_encoder_only(model)
    return train_translation(model) if action == "train" else evaluate_translation(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=["train", "evaluate"])
    args = parser.parse_args()
    run(args.action)
