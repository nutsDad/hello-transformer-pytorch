"""Interactive inference for translation and decoder-only language modeling."""

import torch
from torch.nn.utils.rnn import pad_sequence

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
            config.decoder_only_context_length,
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


def _generation_options(max_new_tokens, temperature, top_k, do_sample):
    max_new_tokens = config.decoder_only_max_new_tokens if max_new_tokens is None else max_new_tokens
    temperature = config.decoder_only_temperature if temperature is None else temperature
    top_k = config.decoder_only_top_k if top_k is None else top_k
    do_sample = config.decoder_only_do_sample if do_sample is None else do_sample
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if top_k is not None and top_k < 0:
        raise ValueError("top_k must be non-negative or None")
    return max_new_tokens, temperature, top_k, do_sample


def _sampling_generator(seed):
    if seed is None:
        return None
    generator = torch.Generator(device=config.device)
    generator.manual_seed(seed)
    return generator


def _next_tokens(log_probs, temperature, top_k, do_sample, generator):
    scaled = log_probs / temperature
    if top_k is not None and 0 < top_k < scaled.size(-1):
        values, _ = torch.topk(scaled, top_k, dim=-1)
        scaled = scaled.masked_fill(scaled < values[:, [-1]], float("-inf"))
    if do_sample:
        return torch.multinomial(
            torch.softmax(scaled, dim=-1),
            num_samples=1,
            generator=generator,
        )
    return torch.argmax(scaled, dim=-1, keepdim=True)


def generate_batch_tokens(
    prompt_tokens,
    model,
    max_new_tokens=None,
    temperature=None,
    top_k=None,
    do_sample=None,
    stop_token_ids=None,
    seed=None,
):
    """Generate continuations for a padded batch while reusing per-layer KV cache."""
    if prompt_tokens.ndim != 2 or prompt_tokens.size(0) == 0:
        raise ValueError("prompt_tokens must have shape [batch_size, sequence_length].")
    max_new_tokens, temperature, top_k, do_sample = _generation_options(
        max_new_tokens, temperature, top_k, do_sample
    )
    tokens = prompt_tokens.to(config.device)
    if tokens.size(1) + max_new_tokens > model.max_positions:
        raise ValueError(
            f"Prompt length {tokens.size(1)} plus max_new_tokens {max_new_tokens} exceeds "
            f"decoder_only_context_length={model.max_positions}."
        )
    stop_token_ids = set(config.decoder_only_stop_token_ids if stop_token_ids is None else stop_token_ids)
    stop_token_ids.add(config.eos_idx)
    generator = _sampling_generator(seed)
    key_padding_mask = tokens != config.padding_idx
    generated = [[] for _ in range(tokens.size(0))]
    finished = torch.zeros(tokens.size(0), dtype=torch.bool, device=config.device)

    with torch.no_grad():
        attention_mask = model.make_causal_mask(tokens)
        hidden_states, past_key_values = model(
            tokens,
            attention_mask=attention_mask,
            use_cache=True,
        )
        last_positions = key_padding_mask.long().sum(dim=1).sub(1)
        batch_indices = torch.arange(tokens.size(0), device=config.device)
        logits = model.generator(hidden_states[batch_indices, last_positions, :])

        for _ in range(max_new_tokens):
            next_token = _next_tokens(logits, temperature, top_k, do_sample, generator)
            next_ids = next_token.squeeze(1)
            was_finished = finished.clone()
            for index, token_id in enumerate(next_ids.tolist()):
                if not was_finished[index] and token_id not in stop_token_ids:
                    generated[index].append(token_id)
            finished |= torch.tensor(
                [token_id in stop_token_ids for token_id in next_ids.tolist()],
                dtype=torch.bool,
                device=config.device,
            )
            if bool(finished.all()):
                break

            # Finished rows are padded for later cache steps; active rows retain
            # their generated token. The padding mask prevents stale rows from
            # contributing keys to active sequences.
            next_input = next_token.masked_fill(finished.unsqueeze(1), config.padding_idx)
            current_valid = ~finished
            key_padding_mask = torch.cat(
                [key_padding_mask, current_valid.unsqueeze(1)],
                dim=1,
            )
            step_mask = key_padding_mask.unsqueeze(1)
            hidden_states, past_key_values = model(
                next_input,
                attention_mask=step_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = model.generator(hidden_states[:, -1, :])
    return generated


def generate_tokens(prompt_tokens, model, **generation_kwargs):
    """Backward-compatible single-prompt wrapper around batch generation."""
    return generate_batch_tokens(prompt_tokens, model, **generation_kwargs)[0]


def _decoder_only_tokenizer():
    return (
        english_tokenizer_load()
        if config.decoder_only_tokenizer == "english"
        else chinese_tokenizer_load()
    )


def generate_texts(prompts, model, **generation_kwargs):
    """Generate text continuations for a list of prompts in one cached batch."""
    if not prompts or not all(isinstance(prompt, str) for prompt in prompts):
        raise ValueError("prompts must be a non-empty list of strings.")
    if len(prompts) > config.decoder_only_server_max_batch_size:
        raise ValueError(
            f"At most {config.decoder_only_server_max_batch_size} prompts are allowed per request."
        )
    tokenizer = _decoder_only_tokenizer()
    token_rows = [
        torch.tensor(
            [tokenizer.bos_id()]
            + tokenizer.EncodeAsIds(prompt)[: config.decoder_only_context_length - 1],
            dtype=torch.long,
        )
        for prompt in prompts
    ]
    prompt_tokens = pad_sequence(
        token_rows,
        batch_first=True,
        padding_value=tokenizer.pad_id(),
    )
    generated_rows = generate_batch_tokens(prompt_tokens, model, **generation_kwargs)
    return [tokenizer.decode_ids(token_ids) for token_ids in generated_rows]


def one_sentence_generate(prompt, model=None, **generation_kwargs):
    """Generate a text continuation from one prompt using the configured LM."""
    if config.model_architecture != "decoder_only":
        raise RuntimeError("Set model_architecture='decoder_only' before generation.")
    model = build_inference_model() if model is None else model
    return generate_texts([prompt], model, **generation_kwargs)[0]


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
