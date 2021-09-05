import torch
import torch.nn as nn
import copy
import sys
import torch.nn.functional as F


class FinetuningWrapper(nn.Module):
    def __init__(self, net,
                 num_classes,
                 seq_len,
                 hidden_dim=100,
                 state_dict=None,
                 weight=None,
                 load_from_pretrained=False,
                 value_guided='plain',
                 clf_reduce='flatten'
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.weight = weight.to(torch.float) if weight is not None else weight
        self.net = copy.deepcopy(net)  # deepcopy is necessary here if we don't want to update the original also
        self.max_seq_len = net.max_seq_len
        self.seq_len = seq_len
        self.load_from_pretrained = load_from_pretrained
        self.value_guided = value_guided
        self.clf_reduce = clf_reduce

        # initialise net hparams from pretrained

        if self.load_from_pretrained and state_dict is not None:
            self.load_state_dict(state_dict)

        # define classifier head layers  # TODO make this more easily customisable

        if clf_reduce == 'flatten':
            num_features = net.attn_layers.dim * self.seq_len
        elif clf_reduce == 'sum':
            num_features = net.attn_layers.dim
        else:
            raise Exception(f"clf_reduce option {clf_reduce} is not implemented!")
        del self.net.to_logits

        if self.value_guided[0:3] == 'vg2':
            if clf_reduce == 'flatten':
                num_guide_ft = net.attn_layers.dim_guide * self.seq_len
            elif clf_reduce == 'sum':
                num_guide_ft = net.attn_layers.dim_guide
            else:
                raise Exception(f"clf_reduce option {clf_reduce} is not implemented!")
            del self.net.to_qlogits
            self.clf = nn.Sequential(
                nn.Linear(num_guide_ft + num_features, hidden_dim, bias=True),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes, bias=True)
            )
        else:
            self.clf = nn.Sequential(
                nn.Linear(num_features, hidden_dim, bias=True),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes, bias=True)
            )

    def forward(self, x, predict=False, **kwargs):  # todo: how should I implement for quantiles_out?
        if self.value_guided == 'plain':
            x, targets = x
            out = self.net(x, return_embeddings=True, **kwargs)
        elif self.value_guided[0:3] == 'vg1':
            x, quantiles, targets = x
            out = self.net(x, quantiles=quantiles, return_embeddings=True, **kwargs)
        elif self.value_guided[0:3] == 'vg2':
            x, quantiles, targets = x
            out, quantiles_out = self.net(x, quantiles=quantiles, return_embeddings=True, **kwargs)

        if self.clf_reduce == 'flatten':
            out = torch.flatten(out, start_dim=1)  # first dim is batch
        elif self.clf_reduce == 'sum':
            out = torch.sum(out, dim=1)

        if self.value_guided[0:3] == 'vg2':
            if self.clf_reduce == 'flatten':
                quantiles_out = torch.flatten(quantiles_out, start_dim=1)  # first dim is batch
            elif self.clf_reduce == 'sum':
                quantiles_out = torch.sum(quantiles_out, dim=1)
            out = torch.cat([out, quantiles_out], dim=1)

        logits = self.clf(out)

        loss = F.cross_entropy(logits, targets, weight=self.weight)  # note: weighted mean, normalised by tot weight.
        return logits if predict else loss
