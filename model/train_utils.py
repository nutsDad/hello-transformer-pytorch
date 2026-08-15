"""Device-agnostic loss and optimizer helpers."""

import torch


class TokenLossCompute:
    """Loss for decoder token prediction on CPU, one GPU, or DataParallel."""

    def __init__(self, generator, criterion, optimizer=None):
        self.generator = generator
        self.criterion = criterion
        self.optimizer = optimizer

    def __call__(self, decoder_output, targets, normalize):
        logits = self.generator(decoder_output)
        raw_loss = self.criterion(
            logits.contiguous().view(-1, logits.size(-1)),
            targets.contiguous().view(-1),
        )
        normalized_loss = raw_loss / normalize.clamp_min(1).to(raw_loss.dtype)
        if self.optimizer is not None:
            self.optimizer.optimizer.zero_grad(set_to_none=True)
            normalized_loss.backward()
            self.optimizer.step()
        return raw_loss.detach().item()


class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for group in self.optimizer.param_groups:
            group["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        step = self._step if step is None else step
        return self.factor * (
            self.model_size ** -0.5
            * min(step ** -0.5, step * self.warmup ** -1.5)
        )


def get_std_opt(model, factor=1.0, warmup=10000):
    """Create the original Noam optimizer for any model in this project."""
    model_size = getattr(model, "d_model", None)
    if model_size is None:
        model_size = model.src_embed[0].d_model
    return NoamOpt(
        model_size,
        factor,
        warmup,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9),
    )
