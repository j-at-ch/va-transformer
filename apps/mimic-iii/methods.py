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
        for i in tqdm.tqdm(range(num_batches), mininterval=0.1, desc=f'epoch {epoch}:', colour='green'):
            for __ in range(batch_size):
                loss = self.model(next(train_loader))
                loss.backward()
            optim.step()
            optim.zero_grad()

            self.writer.add_scalar('train_loss', loss.item(), epoch * num_batches + i)

    @torch.no_grad()
    def evaluate(self, val_loader, epoch, num_batches, batch_size):
        self.model.eval()
        for i in tqdm.tqdm(range(num_batches), mininterval=0.1, desc=f'epoch {epoch}:', colour='green'):
            for __ in range(batch_size):
                loss = self.model(next(val_loader))

            self.writer.add_scalar('train_loss', loss.item(), epoch * num_batches + i)
        # TODO print loss here.


class TuningMethods:
    def __init__(self, model):
        self.model = model


