import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from torch.nn.functional import relu


class MyNetwork(nn.Module):
    def __init__(self, outputs=None, L2_penalty=0, learning_rate=0.01):
        super().__init__()

        self.layer0 = nn.Linear(1, 10)
        self.layer1 = nn.Linear(10, 1)

        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), learning_rate, weight_decay=L2_penalty) # weight_decay is the alpha in weight penalty regularization

        self.to(device)

    def forward(self, x):
        x = x.reshape(-1, 1).float()
        x = self.layer0(x)
        x = relu(x)

        x = self.layer1(x)

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
            losses = []
            n_correct = 0
            self.train()
            for x,y in train_loader:
                train_n += len(x)
                y = y.float().reshape(-1, 1).to(device)
                pred = self.forward(x.to(device))
                loss = self.backward(pred, y)

                losses.append(loss.item())
                n_correct += (pred.reshape(-1) == y).sum().item()


            train_accuracy = n_correct / train_n
            train_avg_loss = sum(losses) / len(losses)
            self.eval()
            val_accuracy, val_avg_loss = self.eval_(val_loader)
            
            if not (dict_of_loss_and_accuracy is None):
                dict_of_loss_and_accuracy['training_loss'].append(train_avg_loss)
                dict_of_loss_and_accuracy['validation_loss'].append(val_avg_loss)
                dict_of_loss_and_accuracy['training_accuracy'].append(train_accuracy)
                dict_of_loss_and_accuracy['validation_accuracy'].append(val_accuracy)
                if save_best and lowest_val_loss > val_avg_loss:
                    torch.save(self.state_dict(), "network")

            print("After epoch: {}\tVal_avg_loss: {:.3f}\tTrain_avg_loss: {:.3f}\tVal_accuracy: {:.3f}\tTrain_accuracy: {:.3f}".format(i, val_avg_loss, train_avg_loss, val_accuracy, train_accuracy))

    def eval_(self, val_data_loader): 
        losses = []
        n_correct = 0
        n_data = 0
        with torch.no_grad():
            for x, y in val_data_loader:
                n_data += len(x)
                y = y.float().reshape(-1, 1).to(device)
                pred = self(x.to(device)) 
                loss = self.loss_fn(pred, y)
                losses.append(loss.item())

                n_correct += torch.sum(pred.reshape(-1) == y).item()
            val_accuracy = n_correct/n_data
            val_avg_loss = sum(losses)/len(losses)    
        
            print(f"x: {x}, pred: {pred.reshape(-1)}")

        return val_accuracy, val_avg_loss


class MyIterAbleDataSet(torch.utils.data.IterableDataset):
    def __init__(self, start, end):
        super(MyIterAbleDataSet).__init__()
        assert end > start, "this code only works with end >= start"
        self.start = start
        self.end = end

    def __iter__(self):
        assert torch.utils.data.get_worker_info().num_workers == 1,  "Only allow single-process data loading"

        return iter([(x, x+1) for x in range(self.start, self.end)])


device = 'cpu'

batch_size = 32
num_workers = 1
learning_rate = 0.001
n_epochs = 50



network = MyNetwork()

train_loader = DataLoader(dataset=MyIterAbleDataSet(0, 10), batch_size=batch_size, num_workers=num_workers)
val_loader = DataLoader(dataset=MyIterAbleDataSet(0, 10), batch_size=batch_size, num_workers=num_workers)

dict_of_loss_and_accuracy = {'training_loss':[], 'validation_loss':[], 'training_accuracy':[], 'validation_accuracy':[]}
network.train_(train_loader, val_loader, n_epochs, dict_of_loss_and_accuracy=dict_of_loss_and_accuracy)

