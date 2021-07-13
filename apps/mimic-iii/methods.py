import numpy as np
import tqdm
import torch


class TrainingMethods:
    def __init__(self, model, writer):
        self.model = model
        clip_value = 0.5
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
        self.writer = writer

    def train(self, train_loader, optim, epoch, num_batches, batch_size):
        self.model.train()
        for i in tqdm.tqdm(range(num_batches), mininterval=0.1, desc=f'epoch {epoch}:'):
            for __ in range(batch_size):
                loss = self.model(next(train_loader))
                loss.backward()
            optim.step()
            optim.zero_grad()

            self.writer.add_scalar('train_loss', loss.item(), epoch * num_batches + i)

    @torch.no_grad()
    def evaluate(self, val_loader, epoch, num_batches, batch_size):
        self.model.eval()
        cum_loss = 0
        for i in tqdm.tqdm(range(num_batches), mininterval=0.1, desc=f'epoch {epoch}:'):
            for __ in range(batch_size):
                loss = self.model(next(val_loader))
                cum_loss += loss.item()
            self.writer.add_scalar('train_loss', loss.item(), epoch * num_batches + i)
        avg_cum_loss = cum_loss/(num_batches*batch_size)
        return avg_cum_loss


class TuningMethods:
    def __init__(self, model):
        self.model = model
