import torch
import torch.nn as nn
import copy
import torch.nn.functional as F


class FinetuningWrapper(nn.Module):
    def __init__(self, net, num_classes, state_dict=None,
                 ignore_index=-100, pad_value=0, weight=None):
        super().__init__()
        self.pad_value = pad_value
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.weight = weight.to(torch.float) if weight is not None else weight
        self.net = copy.deepcopy(net)  # deepcopy is necessary here.
        self.max_seq_len = self.net.max_seq_len

        # initialise net from pretrained

        if state_dict is not None:
            self.net.load_state_dict(state_dict)

        # define classifier head layers

        self.num_features = net.to_logits.in_features * 200
        self.net.clf1 = nn.Linear(self.num_features, num_classes, bias=True)

    def forward(self, X, Y, predict=False, **kwargs):
        Z = self.net(X, return_embeddings=True, **kwargs)
        Z = torch.flatten(Z, start_dim=1)
        logits = self.net.clf1(Z)
        loss = F.cross_entropy(logits, Y, weight=self.weight)
        return logits if predict else loss
