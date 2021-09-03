# The code below relies heavily on the fantastic work from lucidrains in the x_transformers repo:
#       https://github.com/lucidrains/x-transformers/blob/main/x_transformers
import sys
from functools import partial
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from entmax import entmax_bisect


# nucleus

def top_p(logits, thres=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cum_probs > (1 - thres)
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    sorted_logits[sorted_indices_to_remove] = float('-inf')
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)


# topk

def top_k(logits, thres=0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs


# entmax

ENTMAX_ALPHA = 1.3
entmax = entmax_bisect


class AutoregressiveWrapper(nn.Module):
    def __init__(self, net, value_guided=False, ignore_index=-100, pad_value=0, ignore_quantile_index=None):
        super().__init__()
        self.pad_value = pad_value
        self.ignore_index = ignore_index
        self.ignore_quantile_index = ignore_quantile_index
        self.net = net
        self.value_guided = value_guided
        self.max_seq_len = net.max_seq_len

    @torch.no_grad()
    def generate(self, start_tokens, seq_len, eos_token=None, temperature=1., filter_logits_fn=top_k, filter_thres=0.9,
                 **kwargs):
        device = start_tokens.device
        was_training = self.net.training
        num_dims = len(start_tokens.shape)

        if num_dims == 1:
            start_tokens = start_tokens[None, :]

        b, t = start_tokens.shape

        self.net.eval()
        out = start_tokens
        mask = kwargs.pop('mask', None)

        if mask is None:
            mask = torch.full_like(out, True, dtype=torch.bool, device=out.device)

        for _ in range(seq_len):
            x = out[:, -self.max_seq_len:]
            mask = mask[:, -self.max_seq_len:]

            logits = self.net(x, mask=mask, **kwargs)[:, -1, :]

            if filter_logits_fn in {top_k, top_p}:
                filtered_logits = filter_logits_fn(logits, thres=filter_thres)
                probs = F.softmax(filtered_logits / temperature, dim=-1)

            elif filter_logits_fn is entmax:
                probs = entmax(logits / temperature, alpha=ENTMAX_ALPHA, dim=-1)

            sample = torch.multinomial(probs, 1)

            out = torch.cat((out, sample), dim=-1)
            mask = F.pad(mask, (0, 1), value=True)

            if eos_token is not None and (sample == eos_token).all():
                break

        out = out[:, t:]

        if num_dims == 1:
            out = out.squeeze(0)

        self.net.train(was_training)
        return out

    def forward(self, x, **kwargs):

        # help auto-solve a frequent area of confusion around input masks in auto-regressive
        # if user supplies a mask that is only off by one from the source sequence, resolve it for them
        mask = kwargs.get('mask', None)
        if mask is not None and mask.shape[1] == x.shape[1]:
            mask = mask[:, :-1]
            kwargs['mask'] = mask

        if self.value_guided == 'plain':
            xi = x[:, :-1]
            xo = x[:, 1:]
            out = self.net(xi, **kwargs)
        elif self.value_guided[0:3] == 'vg1':
            xi = x[0][:, :-1]
            qi = x[1][:, :-1]
            xo = x[0][:, 1:]
            out = self.net(xi, quantiles=qi, **kwargs)
        elif self.value_guided[0:3] == 'vg2':
            xi = x[0][:, :-1]
            qi = x[1][:, :-1]
            xo = x[0][:, 1:]
            qo = x[1][:, 1:] + 1  # todo: fix this @ the basic level.
            out, quantiles_out = self.net(xi, quantiles=qi, **kwargs)
            token_loss = F.cross_entropy(out.transpose(1, 2), xo, ignore_index=self.ignore_index)
            quantile_loss = F.cross_entropy(quantiles_out.transpose(1, 2), qo, ignore_index=self.ignore_quantile_index)
            return token_loss, quantile_loss

        loss = F.cross_entropy(out.transpose(1, 2), xo, ignore_index=self.ignore_index)  # NOTE: reduction="mean"
        return loss
