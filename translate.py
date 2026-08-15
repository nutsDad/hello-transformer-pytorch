"""Interactive inference for translation and decoder-only language modeling."""

import torch

import config
from beam_decoder import beam_search
from model.checkpoint_utils import load_checkpoint
from model.tf_model import make_decoder_only_model, make_model
from tools.tokenizer_utils import chinese_tokenizer_load, english_tokenizer_load


def build_inference_model(checkpoint_path=None):
    """Build and load the model selected by config.model_architecture."""
    if config.model_architecture == "decoder_only":
        model = make_decoder_only_model(
            config.decoder_only_vocab_size,
            config.n_layers,
            config.d_model,
            config.d_ff,
            config.n_heads,
            config.dropout,
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


def generate_tokens(prompt_tokens, model, max_new_tokens=None, temperature=None, top_k=None, do_sample=None):
    """Generate continuation token IDs with greedy or top-k sampling decoding."""
    max_new_tokens = config.decoder_only_max_new_tokens if max_new_tokens is None else max_new_tokens
    temperature = config.decoder_only_temperature if temperature is None else temperature
    top_k = config.decoder_only_top_k if top_k is None else top_k
    do_sample = config.decoder_only_do_sample if do_sample is None else do_sample
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    tokens = prompt_tokens.to(config.device)
    generated = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            hidden_states = model(tokens)
            log_probs = model.generator(hidden_states[:, -1, :]) / temperature
            if top_k is not None and top_k > 0 and top_k < log_probs.size(-1):
                values, _ = torch.topk(log_probs, top_k, dim=-1)
                log_probs = log_probs.masked_fill(log_probs < values[:, [-1]], float("-inf"))
            if do_sample:
                next_token = torch.multinomial(torch.softmax(log_probs, dim=-1), num_samples=1)
            else:
                next_token = torch.argmax(log_probs, dim=-1, keepdim=True)
            token_id = int(next_token.item())
            if token_id == config.eos_idx:
                break
            generated.append(token_id)
            tokens = torch.cat([tokens, next_token], dim=1)
    return generated


def one_sentence_generate(prompt, model=None, **generation_kwargs):
    """Generate a text continuation from a prompt using the configured LM."""
    if config.model_architecture != "decoder_only":
        raise RuntimeError("Set model_architecture='decoder_only' before generation.")
    model = build_inference_model() if model is None else model
    tokenizer = (
        english_tokenizer_load()
        if config.decoder_only_tokenizer == "english"
        else chinese_tokenizer_load()
    )
    prompt_ids = tokenizer.EncodeAsIds(prompt)[: max(config.decoder_only_max_sequence_length - 1, 1)]
    prompt_tokens = torch.tensor(
        [[tokenizer.bos_id()] + prompt_ids], dtype=torch.long, device=config.device
    )
    generated_ids = generate_tokens(prompt_tokens, model, **generation_kwargs)
    return tokenizer.decode_ids(generated_ids)


def interactive_example():
    model = build_inference_model()
    while True:
        sentence = input("Input text (empty line exits): ").strip()
        if not sentence:
            return
        if config.model_architecture == "encoder_decoder":
            print(one_sentence_translate(sentence, model))
        else:
            print(one_sentence_generate(sentence, model))


if __name__ == "__main__":
    interactive_example()
