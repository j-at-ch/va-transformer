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
        cum_loss = 0
        for i in tqdm.tqdm(range(num_batches), mininterval=0.1, desc=f'epoch {epoch}'):
            batch_loss = 0
            for __ in range(batch_size):
                loss = self.model(next(train_loader))
                loss.backward()
                batch_loss += loss.item()
            optim.step()
            optim.zero_grad()
            #print(f'avg batch loss: {batch_loss/batch_size}')
            self.writer.add_scalar('train_loss', batch_loss/batch_size, epoch * num_batches + i)
            cum_loss += batch_loss
        avg_loss = cum_loss / (num_batches * batch_size)
        print(f'epoch avg train loss: {avg_loss}')
        return avg_loss

    @torch.no_grad()
    def evaluate(self, val_loader, epoch, num_batches, batch_size):
        self.model.eval()
        cum_loss = 0
        for i in tqdm.tqdm(range(num_batches), mininterval=0.1, desc=f'epoch {epoch}'):
            for __ in range(batch_size):
                loss = self.model(next(val_loader))
                cum_loss += loss.item()
        avg_val_loss = cum_loss / (num_batches * batch_size)
        self.writer.add_scalar('avg_val_loss', avg_val_loss, epoch * num_batches + i)
        return avg_val_loss


class FinetuningMethods:
    def __init__(self, model, writer):
        self.model = model
        clip_value = 0.5
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
        self.writer = writer

    def train(self, train_loader, optim, epoch, num_batches, batch_size):
        self.model.train()
        cum_loss = 0
        for i in tqdm.tqdm(range(num_batches), mininterval=0.1, desc=f'epoch {epoch}'):
            batch_loss = 0
            for __ in range(batch_size):
                X, Y = next(train_loader)
                loss = self.model(X, Y)
                loss.backward()
                batch_loss += loss.item()
            optim.step()
            optim.zero_grad()
            #print(f'avg batch loss: {batch_loss/batch_size}')
            self.writer.add_scalar('train_loss', batch_loss/batch_size, epoch * num_batches + i)
            cum_loss += batch_loss
        avg_loss = cum_loss / (num_batches * batch_size)
        print(f'epoch avg train loss: {avg_loss}')
        return avg_loss

    @torch.no_grad()
    def evaluate(self, val_loader, epoch, num_batches, batch_size):
        self.model.eval()
        cum_loss = 0
        for i in tqdm.tqdm(range(num_batches), mininterval=0.1, desc=f'epoch {epoch}'):
            for __ in range(batch_size):
                loss = self.model(next(val_loader))
                cum_loss += loss.item()
        avg_val_loss = cum_loss / (num_batches * batch_size)
        self.writer.add_scalar('avg_val_loss', avg_val_loss, epoch * num_batches + i)
        return avg_val_loss