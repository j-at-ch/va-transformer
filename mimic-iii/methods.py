import sys
import numpy as np
import tqdm
import torch


class TrainingMethods:
    def __init__(self, model, writer):
        self.model = model
        clip_value = 0.5
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
        self.writer = writer

    def train(self, train_loader, optim, epoch, batch_size):
        self.model.train()
        cum_loss = 0
        for i, X in tqdm.tqdm(enumerate(train_loader), total=len(train_loader),
                              mininterval=0.5, desc=f'epoch {epoch} training'):
            loss = self.model(X)
            loss.backward()
            batch_loss = loss.item()
            optim.step()
            optim.zero_grad()
            self.writer.add_scalar('train_loss', batch_loss / batch_size, epoch * len(train_loader) + i)
            cum_loss += batch_loss
        avg_loss = cum_loss / (len(train_loader) * batch_size)
        print(f'epoch avg train loss: {avg_loss}')
        return avg_loss

    @torch.no_grad()
    def evaluate(self, val_loader, epoch, batch_size):
        self.model.eval()
        cum_loss = 0
        for i, X in tqdm.tqdm(enumerate(val_loader), total=len(val_loader),
                              mininterval=0.5, desc=f'epoch {epoch} evaluation'):
            loss = self.model(X)
            cum_loss += loss.item()
        avg_loss = cum_loss / (len(val_loader) * batch_size)
        self.writer.add_scalar('avg_val_loss', avg_loss, epoch * len(val_loader) + i)
        print(f'epoch avg val loss: {avg_loss}')
        return avg_loss


class FinetuningMethods:
    def __init__(self, model, writer):
        self.model = model
        clip_value = 0.5
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
        self.writer = writer

    def train(self, train_loader, optim, epoch, num_batches, batch_size):
        self.model.train()
        cum_loss = 0
        for i in tqdm.tqdm(range(num_batches), mininterval=1, desc=f'epoch {epoch} training'):
            batch_loss = 0
            for __ in range(batch_size):
                X, Y = next(train_loader)
                loss = self.model(X, Y)
                loss.backward()
                batch_loss += loss.item()
            optim.step()
            optim.zero_grad()
            # print(f'avg batch loss: {batch_loss/batch_size}')
            self.writer.add_scalar('loss/train', batch_loss / batch_size, epoch * num_batches + i)
            cum_loss += batch_loss
        avg_loss = cum_loss / (num_batches * batch_size)
        self.writer.add_scalar('finetuning/loss/train', avg_loss, (epoch + 1) * num_batches + i)
        print(f'epoch avg train loss: {avg_loss}')
        return avg_loss

    @torch.no_grad()
    def evaluate(self, val_loader, epoch, num_batches, batch_size):
        self.model.eval()
        cum_loss = 0
        for i in tqdm.tqdm(range(num_batches), mininterval=0.1, desc=f'epoch {epoch} evaluation'):
            batch_loss = 0
            for _ in range(batch_size):
                X, Y = next(val_loader)
                loss = self.model(X, Y)
                batch_loss += loss.item()
            self.writer.add_scalar('loss/val', batch_loss / batch_size, epoch * num_batches + i)
            cum_loss += batch_loss
        avg_loss = cum_loss / (num_batches * batch_size)
        self.writer.add_scalar('finetuning/loss/eval', avg_loss, (epoch + 1) * num_batches)
        print(f'epoch avg val loss: {avg_loss}')
        return avg_loss
