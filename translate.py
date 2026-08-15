"""Interactive inference for both configured Transformer architectures."""

import torch

import config
from beam_decoder import beam_search
from model.checkpoint_utils import load_checkpoint
from model.tf_model import make_encoder_only_model, make_model
from tools.tokenizer_utils import chinese_tokenizer_load, english_tokenizer_load


def build_inference_model(checkpoint_path=None):
    """Build and load the model selected by config.model_architecture."""
    if config.model_architecture == "encoder_only":
        model = make_encoder_only_model(
            config.encoder_only_vocab_size,
            config.encoder_only_num_labels,
            config.n_layers,
            config.d_model,
            config.d_ff,
            config.n_heads,
            config.dropout,
            config.encoder_only_pooling,
        )
    else:
        model = make_model(
            config.src_vocab_size,
            config.tgt_vocab_size,
            config.n_layers,
            config.d_model,
            config.d_ff,
            config.n_heads,
            config.dropout,
        )
    load_checkpoint(
        checkpoint_path or config.inference_model_path,
        model,
        config.device,
        config.model_architecture,
    )
    return model.eval()


def _input_tokens(sentence, tokenizer):
    token_ids = tokenizer.EncodeAsIds(sentence)[: max(config.max_source_len - 2, 0)]
    return torch.tensor(
        [[tokenizer.bos_id()] + token_ids + [tokenizer.eos_id()]],
        dtype=torch.long,
        device=config.device,
    )


def translate(src, model):
    """Translate a pre-tokenized English source sentence with a loaded model."""
    if config.model_architecture != "encoder_decoder":
        raise RuntimeError("translate() requires model_architecture='encoder_decoder'.")
    with torch.no_grad():
        src_mask = (src != config.padding_idx).unsqueeze(-2)
        decoded, _ = beam_search(
            model,
            src,
            src_mask,
            config.max_len,
            config.padding_idx,
            config.bos_idx,
            config.eos_idx,
            config.beam_size,
            config.device,
        )
        return chinese_tokenizer_load().decode_ids(decoded[0][0])


def one_sentence_translate(sentence, model=None):
    """Translate one English sentence using the configured checkpoint."""
    if config.model_architecture != "encoder_decoder":
        raise RuntimeError("Set model_architecture='encoder_decoder' before translation.")
    model = build_inference_model() if model is None else model
    return translate(_input_tokens(sentence, english_tokenizer_load()), model)


def one_sentence_predict(sentence, model=None):
    """Return encoder-only class ID, optional label name, and probabilities."""
    if config.model_architecture != "encoder_only":
        raise RuntimeError("Set model_architecture='encoder_only' before classification.")
    model = build_inference_model() if model is None else model
    tokenizer = english_tokenizer_load() if config.encoder_only_tokenizer == "english" else chinese_tokenizer_load()
    src = _input_tokens(sentence, tokenizer)
    with torch.no_grad():
        logits = model(src, (src != config.padding_idx).unsqueeze(-2))
        probabilities = torch.softmax(logits, dim=-1)[0].cpu().tolist()
        label_id = int(torch.argmax(logits, dim=-1).item())
    label_name = (
        config.encoder_only_label_names[label_id]
        if label_id < len(config.encoder_only_label_names)
        else str(label_id)
    )
    return {"label_id": label_id, "label": label_name, "probabilities": probabilities}


def one_sentence_encode(sentence, model=None):
    """Return the encoder-only pooled sentence embedding as a Python list."""
    if config.model_architecture != "encoder_only":
        raise RuntimeError("Set model_architecture='encoder_only' before embedding inference.")
    model = build_inference_model() if model is None else model
    tokenizer = english_tokenizer_load() if config.encoder_only_tokenizer == "english" else chinese_tokenizer_load()
    src = _input_tokens(sentence, tokenizer)
    with torch.no_grad():
        vector = model(src, (src != config.padding_idx).unsqueeze(-2), return_embeddings=True)
    return vector[0].cpu().tolist()


def interactive_example():
    model = build_inference_model()
    while True:
        sentence = input("Input text (empty line exits): ").strip()
        if not sentence:
            return
        if config.model_architecture == "encoder_decoder":
            print(one_sentence_translate(sentence, model))
        else:
            print(one_sentence_predict(sentence, model))


if __name__ == "__main__":
    interactive_example()
