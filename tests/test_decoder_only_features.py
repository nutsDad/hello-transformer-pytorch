"""Regression tests for the decoder-only training and inference additions."""

import tempfile
import unittest
from pathlib import Path

import torch

import config
from model.checkpoint_utils import restore_training_checkpoint, save_checkpoint
from model.tf_model import make_decoder_only_model
from model.train_utils import get_std_opt
from tools.data_loader import clean_decoder_only_texts
from tools.prepare_lm_data import split_items
from translate import generate_batch_tokens


class DecoderOnlyFeatureTests(unittest.TestCase):
    def make_model(self):
        torch.manual_seed(7)
        return make_decoder_only_model(
            vocab_size=32,
            N=2,
            d_model=16,
            d_ff=32,
            h=4,
            dropout=0.0,
            max_positions=32,
        ).eval()

    def test_cleaning_and_split_are_deterministic(self):
        cleaned, statistics = clean_decoder_only_texts(
            ["  hello   world ", "", "x", "hello world", "another document"],
            min_characters=2,
            deduplicate=True,
        )
        self.assertEqual(cleaned, ["hello world", "another document"])
        self.assertEqual(statistics["dropped_empty"], 1)
        self.assertEqual(statistics["dropped_too_short"], 1)
        self.assertEqual(statistics["dropped_duplicates"], 1)

        items = [f"document {index}" for index in range(10)]
        self.assertEqual(
            split_items(items, 0.2, 0.2, seed=11),
            split_items(items, 0.2, 0.2, seed=11),
        )

    def test_kv_cache_matches_full_forward(self):
        model = self.make_model()
        tokens = torch.tensor([[2, 5, 6, 7]], dtype=torch.long)
        with torch.no_grad():
            full_hidden = model(tokens)
            _, cache = model(tokens[:, :3], use_cache=True)
            cached_hidden, _ = model(tokens[:, 3:], past_key_values=cache, use_cache=True)
        self.assertTrue(torch.allclose(full_hidden[:, 3:], cached_hidden, atol=1e-6))

    def test_seeded_batch_generation_is_repeatable(self):
        model = self.make_model()
        prompts = torch.tensor([[2, 5, 6], [2, 7, 0]], dtype=torch.long)
        first = generate_batch_tokens(
            prompts,
            model,
            max_new_tokens=3,
            top_k=8,
            do_sample=True,
            seed=123,
        )
        second = generate_batch_tokens(
            prompts,
            model,
            max_new_tokens=3,
            top_k=8,
            do_sample=True,
            seed=123,
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)

    def test_checkpoint_restores_optimizer_schedule(self):
        model = self.make_model().train()
        optimizer = get_std_opt(model)
        tokens = torch.tensor([[2, 5, 6]], dtype=torch.long)
        targets = torch.tensor([[5, 6, 3]], dtype=torch.long)
        hidden = model(tokens)
        loss = torch.nn.functional.nll_loss(
            model.generator(hidden).reshape(-1, 32),
            targets.reshape(-1),
        )
        optimizer.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "resume.pth"
            save_checkpoint(
                checkpoint,
                model,
                "decoder_only",
                optimizer=optimizer,
                epoch=1,
            )
            restored_model = self.make_model().train()
            restored_optimizer = get_std_opt(restored_model)
            metadata = restore_training_checkpoint(
                checkpoint,
                restored_model,
                config.device,
                "decoder_only",
                restored_optimizer,
            )
        self.assertEqual(metadata["epoch"], 1)
        self.assertEqual(restored_optimizer._step, optimizer._step)


if __name__ == "__main__":
    unittest.main()
