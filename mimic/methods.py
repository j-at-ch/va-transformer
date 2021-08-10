import tqdm
import torch


class TrainingMethods:
    def __init__(self, model, writer):
        self.model = model
        clip_value = 0.5
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
        self.writer = writer

    def train(self, train_loader, optim, epoch):
        self.model.train()
        cum_loss = 0
        for i, X in tqdm.tqdm(enumerate(train_loader), total=len(train_loader),
                              mininterval=0.5, desc=f'epoch {epoch} training'):
            loss = self.model(X)
            loss.backward()
            batch_loss = loss.item()
            optim.step()
            optim.zero_grad()
            self.writer.add_scalar('loss/train', batch_loss, epoch * len(train_loader) + i)
            cum_loss += batch_loss

        epoch_loss = cum_loss / len(train_loader)
        print(f'epoch avg train loss: {epoch_loss}')
        return epoch_loss

    @torch.no_grad()
    def evaluate(self, val_loader, epoch):
        self.model.eval()
        cum_loss = 0
        for i, X in tqdm.tqdm(enumerate(val_loader), total=len(val_loader),
                              mininterval=0.5, desc=f'epoch {epoch} evaluation'):
            loss = self.model(X)
            cum_loss += loss.item()

        epoch_loss = cum_loss / len(val_loader)
        self.writer.add_scalar('loss/val', epoch_loss, (epoch + 1) * len(val_loader))
        print(f'epoch avg val loss: {epoch_loss}')
        return epoch_loss


class FinetuningMethods:  # NOTE: FineTuning and Training are now equal except for the '*' in self.model.
    def __init__(self, model, writer):
        self.model = model
        clip_value = 0.5
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
        self.writer = writer

    def train(self, train_loader, optim, epoch):
        self.model.train()
        cum_loss = 0
        for i, X in tqdm.tqdm(enumerate(train_loader), total=len(train_loader),
                              mininterval=0.5, desc=f'epoch {epoch} training'):
            loss = self.model(*X)
            loss.backward()
            batch_loss = loss.item()
            optim.step()
            optim.zero_grad()
            self.writer.add_scalar('loss/train', batch_loss, epoch * len(train_loader) + i)
            cum_loss += batch_loss

        epoch_loss = cum_loss / len(train_loader)
        print(f'epoch avg train loss: {epoch_loss}')
        return epoch_loss

    @torch.no_grad()
    def evaluate(self, val_loader, epoch):
        self.model.eval()
        cum_loss = 0
        for i, X in tqdm.tqdm(enumerate(val_loader), total=len(val_loader),
                              mininterval=0.5, desc=f'epoch {epoch} evaluation'):
            loss = self.model(*X)
            cum_loss += loss.item()

        epoch_loss = cum_loss / len(val_loader)
        self.writer.add_scalar('loss/val', epoch_loss, (epoch + 1) * len(val_loader))
        print(f'epoch avg val loss: {epoch_loss}')
        return epoch_loss
