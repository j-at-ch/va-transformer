import torch
import torch.nn as nn
import copy
import sys
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, num_in, num_hidden, num_out, dropout_hidden):
        super().__init__()
        self.num_in = num_in
        self.num_hidden = num_hidden
        self.num_out = num_out
        self.dropout_hidden = dropout_hidden
        self.net = nn.Sequential(
            nn.Linear(num_in, num_hidden, bias=True),
            nn.ReLU(),
            nn.Dropout(p=dropout_hidden),
            nn.Linear(num_hidden, num_out, bias=True)
        )

    def forward(self, x):
        return self.net(x)


class FinetuningWrapper(nn.Module):
    def __init__(self, net,
                 num_classes,
                 seq_len,
                 hidden_dim=100,
                 state_dict=None,
                 weight=None,
                 load_from='pretrained',
                 value_guided='plain',
                 clf_style='flatten',
                 clf_dropout=0.
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.weight = weight.to(torch.float) if weight is not None else weight
        self.net = copy.deepcopy(net)
        self.max_seq_len = net.max_seq_len
        self.seq_len = seq_len
        self.load_from = load_from
        self.value_guided = value_guided
        self.clf_style = clf_style
        self.clf_dropout = clf_dropout

        # initialise net hparams from pretrained

        if self.load_from == 'pretrained' and state_dict is not None:
            self.load_state_dict(state_dict)

        # define classifier head layers

        if clf_style == 'flatten':
            num_features = net.attn_layers.dim * self.seq_len
        elif clf_style in ['on_SOS', 'on_EOS', 'sum']:
            num_features = net.attn_layers.dim
        else:
            raise Exception(f"clf_reduce option {clf_style} is not implemented!")
        del self.net.to_logits

        if self.value_guided[0:3] == 'vg2':
            if clf_style == 'flatten':
                num_guide_ft = net.attn_layers.dim_guide * self.seq_len
            elif clf_style in ['on_SOS', 'on_EOS', 'sum']:
                num_guide_ft = net.attn_layers.dim_guide
            else:
                raise Exception(f"clf_reduce option {clf_style} is not implemented!")
            del self.net.to_guide_logits
            self.clf = Classifier(num_guide_ft + num_features, hidden_dim, num_classes, clf_dropout)
        else:
            self.clf = Classifier(num_features, hidden_dim, num_classes, clf_dropout)

        # if doing post-training analysis then initialise net hparams from finetuned model

        if self.load_from == 'finetuned' and state_dict is not None:
            self.load_state_dict(state_dict)

    def forward(self, x, predict=False, **kwargs):
        if self.value_guided == 'plain':
            x, targets = x
            out = self.net(x, return_embeddings=True, **kwargs)
        elif self.value_guided[0:3] == 'vg1':
            x, quantiles, targets = x
            out = self.net(x, quantiles=quantiles, return_embeddings=True, **kwargs)
        elif self.value_guided[0:3] == 'vg2':
            x, quantiles, targets = x
            out, quantiles_out = self.net(x, quantiles=quantiles, return_embeddings=True, **kwargs)

        if self.clf_style == 'flatten':
            out = torch.flatten(out, start_dim=1)  # first dim is batch
        elif self.clf_style == 'sum':
            out = torch.sum(out, dim=1)
        elif self.clf_style == 'on_SOS':
            out = out[:, 0, :]
        elif self.clf_style == 'on_EOS':
            out = out[:, -1, :]

        if self.value_guided[0:3] == 'vg2':
            if self.clf_style == 'flatten':
                quantiles_out = torch.flatten(quantiles_out, start_dim=1)  # first dim is batch
            elif self.clf_style == 'sum':
                quantiles_out = torch.sum(quantiles_out, dim=1)
            elif self.clf_style == 'on_SOS':
                quantiles_out = quantiles_out[:, 0, :]
            elif self.clf_style == 'on_EOS':
                quantiles_out = quantiles_out[:, -1, :]
            out = torch.cat([out, quantiles_out], dim=1)

        logits = self.clf(out)

        loss = F.cross_entropy(logits, targets, weight=self.weight)  # note: weighted mean, normalised by tot weight.
        return logits if predict else loss
