import torch
import torch.nn as nn
import copy
import torch.nn.functional as F


class FinetuningWrapper(nn.Module):
    def __init__(self, net, num_classes, seq_len,
                 state_dict=None, weight=None,
                 load_from_pretuning=False,
                 ignore_index=-100, pad_value=0):
        super().__init__()
        self.pad_value = pad_value
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.weight = weight.to(torch.float) if weight is not None else weight
        self.net = copy.deepcopy(net)  # deepcopy is necessary here if we don't want to update the original also.
        self.max_seq_len = self.net.max_seq_len
        self.seq_len = seq_len
        self.load_from_pretuning = load_from_pretuning

        # initialise net from pretrained

        if self.load_from_pretuning and state_dict is not None:
            self.load_state_dict(state_dict)

        # define classifier head layers

        self.num_features = net.to_logits.in_features * self.seq_len
        self.net.clf1 = nn.Linear(self.num_features, num_classes, bias=True)
        del self.net.to_logits

    def forward(self, X, Y, predict=False, **kwargs):
        Z = self.net(X, return_embeddings=True, **kwargs)
        Z = torch.flatten(Z, start_dim=1)  # consider alternatives?
        #Z = torch.flatten(Z[:, 0, :], start_dim=1)  # TEST
        logits = self.net.clf1(Z)
        loss = F.cross_entropy(logits, Y, weight=self.weight)  # note: weighted mean, normalised by tot weight.
        return logits if predict else loss