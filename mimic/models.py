import torch
import torch.nn as nn
import copy
import torch.nn.functional as F


class FinetuningWrapper(nn.Module):  # TODO: if loading from pretrained, then we don't want to have to specify params.
    def __init__(self, net,
                 num_classes,
                 seq_len,
                 state_dict=None,
                 weight=None,
                 load_from_pretrained=False,
                 value_guided='plain',
                 clf_reduce='flatten',
                 ignore_index=-100,
                 pad_value=0
                 ):
        super().__init__()
        self.pad_value = pad_value
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.weight = weight.to(torch.float) if weight is not None else weight
        self.net = copy.deepcopy(net)  # deepcopy is necessary here if we don't want to update the original also
        self.max_seq_len = net.max_seq_len
        self.seq_len = seq_len
        self.load_from_pretrained = load_from_pretrained
        self.value_guided = value_guided
        self.clf_reduce = clf_reduce

        # initialise net from pretrained

        if self.load_from_pretrained and state_dict is not None:
            self.load_state_dict(state_dict)

        # define classifier head layers  # TODO make this more easily customisable

        if clf_reduce == 'flatten':
            num_features = net.to_logits.in_features * self.seq_len
            del self.net.to_logits
        elif clf_reduce == 'sum':
            num_features = net.to_logits.in_features
            del self.net.to_logits

        self.clf = nn.Linear(num_features, num_classes, bias=True)

        # specify clf  # TODO: tidy/formalise
        #self.clf = nn.ModuleList([
        #    nn.Linear(self.num_features, num_classes, bias=True)
        #])

    def forward(self, x, predict=False, **kwargs):
        if self.value_guided == 'plain':
            x, targets = x
            out = self.net(x, return_embeddings=True, **kwargs)
        else:
            x, quantiles, targets = x
            out = self.net(x, quantiles=quantiles, return_embeddings=True, **kwargs)

        # flatten out vs sum out
        out = torch.flatten(out, start_dim=1)  # first dim is batch
        #out = torch.sum(out, dim=1)
        #Z = torch.flatten(Z[:, 0, :], start_dim=1)
        logits = self.clf(out)

        loss = F.cross_entropy(logits, targets, weight=self.weight)  # note: weighted mean, normalised by tot weight.
        return logits if predict else loss
