import tqdm
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score


class TrainingMethods:
    def __init__(self, model, writer):
        self.model = model
        clip_value = 0.5
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
        self.writer = writer

    def train(self, train_loader, optimizer, epoch):
        self.model.train()
        cum_loss = 0
        for i, X in tqdm.tqdm(enumerate(train_loader), total=len(train_loader),
                              mininterval=0.5, desc=f'epoch {epoch} training'):
            loss = self.model(X)
            loss.backward()
            batch_loss = loss.item()
            optimizer.step()
            optimizer.zero_grad()
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

    @torch.no_grad()
    def write_embeddings(self, step, mappings, labeller, seq_len, device):
        self.model.eval()
        tokens = list(mappings.topNtokens_tr(N=2000).keys())
        x = torch.tensor(tokens, dtype=torch.int)
        z = torch.Tensor().to(device)
        for x_part in torch.split(x, seq_len):
            x_part = x_part.to(device)
            z_part = self.model.net.token_emb(x_part)
            z = torch.cat((z, z_part))
        metadata = [label for label in map(labeller.token2label, x.cpu().numpy())]
        self.writer.add_embedding(z,
                                  metadata=metadata,
                                  global_step=step,
                                  tag='token_embeddings')


class FinetuningMethods:  # NOTE: FineTuning and Training are now equal except for the '*' in self.model and predict.
    def __init__(self, model, writer):
        self.model = model
        clip_value = 0.5
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
        self.writer = writer

    def train(self, train_loader, optimizer, epoch):
        self.model.train()
        cum_loss = 0
        for i, X in tqdm.tqdm(enumerate(train_loader), total=len(train_loader),
                              mininterval=0.5, desc=f'epoch {epoch} training'):
            loss = self.model(*X)
            loss.backward()
            batch_loss = loss.item()
            optimizer.step()
            optimizer.zero_grad()
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

    @torch.no_grad()
    def predict(self, data_loader, epoch, device, prefix="val"):
        self.model.eval()
        y_score = torch.tensor([]).to(device)
        y_true = torch.tensor([]).to(device)
        for i, (x, y) in enumerate(data_loader):
            y_true = torch.cat((y_true, y))
            logits = self.model(x, y, predict=True)
            y_score = torch.cat((y_score, F.softmax(logits, dim=1)))
        y_true = y_true.cpu()
        y_score = y_score.cpu()

        acc = accuracy_score(y_true, torch.argmax(y_score, dim=1), normalize=True)
        bal_acc = balanced_accuracy_score(y_true, torch.argmax(y_score, dim=1))
        roc_auc = roc_auc_score(y_true, y_score[:, 1])

        self.writer.add_scalar(prefix + '/acc', acc, epoch)
        self.writer.add_scalar(prefix + '/bal_acc', bal_acc, epoch)
        self.writer.add_scalar(prefix + '/roc_auc', roc_auc, epoch)
        self.writer.add_pr_curve(prefix + '/pr_curve', y_true, y_score[:, 1], epoch)
        print(f'epoch {prefix}/roc_auc = {roc_auc}, {prefix}/bal_acc = {bal_acc}, {prefix}/acc = {acc}')
        return y_score, y_true

    @torch.no_grad()
    def write_embeddings(self, step, mappings, labeller, seq_len, device):
        self.model.eval()
        tokens = list(mappings.topNtokens_tr(N=2000).keys())
        x = torch.tensor(tokens, dtype=torch.int)
        z = torch.Tensor().to(device)
        for x_part in torch.split(x, seq_len):
            x_part = x_part.to(device)
            z_part = self.model.net.token_emb(x_part)
            z = torch.cat((z, z_part))
        metadata = [label for label in map(labeller.token2label, x.cpu().numpy())]
        self.writer.add_embedding(z,
                                  metadata=metadata,
                                  global_step=step,
                                  tag='token_embeddings')
