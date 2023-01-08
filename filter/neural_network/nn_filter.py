import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from torch.nn.functional import relu
import time

device = 'cpu'

class MyNetwork(nn.Module):
    def __init__(self, outputs=None, L2_penalty=0, learning_rate=0.01):
        super().__init__()

        self.norm0 = nn.BatchNorm1d(10)
        self.layer0 = nn.Linear(10, 32)
        self.dropout0 = nn.Dropout(p=0.2, inplace=False)

        self.norm1 = nn.BatchNorm1d(32)
        self.layer1 = nn.Linear(32, 32)
        self.dropout1 = nn.Dropout(p=0.2, inplace=False)

        self.norm2 = nn.BatchNorm1d(32)
        self.layer2 = nn.Linear(32, 64)

        self.layer3 = nn.Linear(64, 64)
        self.layer4 = nn.Linear(64, 64)
        self.layer5 = nn.Linear(64, 32)
        self.layer6 = nn.Linear(32, 32)
        self.layer7 = nn.Linear(32, 4)

        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), learning_rate, weight_decay=L2_penalty) # weight_decay is the alpha in weight penalty regularization

        self.to(device)

    def forward(self, x):
        x = x.reshape(-1, 10).float()

        # x = self.norm0(x)
        x = self.layer0(x)
        # x = self.dropout0(x)
        x = relu(x)

        # x = self.norm1(x)
        x = self.layer1(x)
        # x = self.dropout1(x)
        x = relu(x)

        # x = self.norm2(x)
        x = self.layer2(x)

        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

        return x

    def backward(self, predicted, truth):
        loss = self.loss_fn(predicted, truth)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss

    def train_(self, train_loader, val_loader, epochs, dict_of_loss_and_accuracy = None, save_best=True):
        lowest_val_loss = 1e10
        train_n = 0
        
        for i in range(epochs):
            t_start = time.time()
            losses = []
            self.train()
            for x,y in train_loader:
                train_n += len(x)
                y = y.float().reshape(-1, 4).to(device)
                pred = self.forward(x.to(device))
                loss = self.backward(pred, y)

                losses.append(loss.item())


            train_avg_loss = sum(losses) / len(losses)
            self.eval()
            val_avg_loss = self.eval_(val_loader)
            train_time = time.time() - t_start
            
            if not (dict_of_loss_and_accuracy is None):
                dict_of_loss_and_accuracy['training_loss'].append(train_avg_loss)
                dict_of_loss_and_accuracy['validation_loss'].append(val_avg_loss)
                dict_of_loss_and_accuracy['train_time'].append(train_time)
                if save_best and lowest_val_loss > val_avg_loss:
                    torch.save(self.state_dict(), "network")

            print("After epoch: {}\tVal_avg_loss: {:.3f}\tTrain_avg_loss: {:.3f}\ttrain_time: {:.2f} s".format(i, val_avg_loss, train_avg_loss, train_time))

    def eval_(self, val_data_loader): 
        losses = []
        n_data = 0
        with torch.no_grad():
            for x, y in val_data_loader:
                n_data += len(x)
                y = y.float().reshape(-1, 4).to(device)
                pred = self(x.to(device)) 
                loss = self.loss_fn(pred, y)
                losses.append(loss.item())

            val_avg_loss = sum(losses)/len(losses)    
        
        return val_avg_loss


if __name__ == '__main__':
    from dataloader import MyIterAbleDataSet
    batch_size = 64
    num_workers = 1
    learning_rate = 0.0005
    n_epochs = 1000

    network = MyNetwork()

    train_loader = DataLoader(dataset=MyIterAbleDataSet('filter/neural_network/train_data'), batch_size=batch_size, num_workers=num_workers)
    val_loader = DataLoader(dataset=MyIterAbleDataSet('filter/neural_network/val_data'), batch_size=batch_size, num_workers=num_workers)

    dict_of_loss_and_accuracy = {'training_loss':[], 'validation_loss':[], 'training_accuracy':[], 'validation_accuracy':[], 'train_time':[]}
    network.train_(train_loader, val_loader, n_epochs, dict_of_loss_and_accuracy=dict_of_loss_and_accuracy)
