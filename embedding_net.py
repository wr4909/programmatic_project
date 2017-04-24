#!/usr/bin/python3

import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=3, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=80, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


df = pd.read_csv('Programmatic Project_Scoring_TTD pixel fires.csv', encoding='latin1', parse_dates=['logentrytime'])

# Remove duplicate tdid
visited_tdids = df[df.trackingtagid.as_matrix() == 'qelg9wq']['tdid']
df['contact'] = df.tdid.isin(visited_tdids)
df = df.sort_values('logentrytime', ascending=False).drop_duplicates('tdid', keep='first')

Y = df['contact'].as_matrix() * 1
categorical_columns = ['country', 'region', 'metro', 'devicetype', 'osfamily', 'browser', 'devicemake']

df['dow'] = df['logentrytime'].dt.dayofweek
df['hour'] = df['logentrytime'].dt.hour
df['month'] = df['logentrytime'].dt.month
df['day'] = df['logentrytime'].dt.day
categorical_columns += ['dow', 'hour', 'month', 'day']

X = np.zeros([df.shape[0], len(categorical_columns)], dtype=np.int64)
col_unique = {c: np.unique(df[c].as_matrix()).shape[0] for c in categorical_columns}
label_encoders = [preprocessing.LabelEncoder() for c in categorical_columns]
for i in range(len(categorical_columns)):
    l = label_encoders[i]
    c = categorical_columns[i]
    X[:,i] = l.fit_transform(df[c].tolist())

X_train, X_val, Y_train, Y_val = (torch.from_numpy(v) for v in train_test_split(X, Y, test_size=0.2, random_state=42))
embedding_sizes = {c: 5 for c in categorical_columns}

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_train, Y_train),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_val, Y_val),
    batch_size=args.batch_size, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self, intermediate_sizes=[]):
        super(Net, self).__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(col_unique[c], embedding_sizes[c]) for c in categorical_columns])
        sizes = [sum(embedding_sizes.values())] + intermediate_sizes + [2]
        self.layers = nn.ModuleList([nn.Linear(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)])

    def forward(self, x):
        x = [self.embeddings[i](torch.unsqueeze(x[:,i], 0))[0] for i in range(len(categorical_columns))]
        x = torch.cat(x, 1)
        for i in range(len(self.layers)-1):
            x = F.relu(self.layers[i](x))
        x = self.layers[-1](x)
        return F.log_softmax(x)

model = Net([45]*4)
if args.cuda:
    model.cuda()

#optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.00001)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

def train(epoch):
    model.train()
    train_weights = torch.from_numpy(np.array([1.0,22.0])).float().cuda()
    criterion = torch.nn.NLLLoss(train_weights)
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data = data.cuda()
            target = target.cuda()
        data = Variable(data)
        target = Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        param_count = sum([torch.numel(p.data) for p in model.parameters()])
        l2 = .2 * sum([torch.sum(p.data**2) for p in model.parameters()])/param_count
        l1 = .2 * sum([torch.sum(torch.abs(p.data)) for p in model.parameters()])/param_count
        reg_loss = loss + l2 + l1 

        reg_loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print("[Epoch %3d] [ %7d / %7d - %2.0f%% ]\tLoss %.6f\tRegLoss %.6f" % (epoch, batch_idx * len(data), len(train_loader.dataset), 100.0 * batch_idx / len(train_loader), loss.data[0], reg_loss.data[0]))

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    true_pos = 0
    all_pos = 0
    for data, target in test_loader:
        if args.cuda:
            data = data.cuda()
            target = target.cuda()
        data = Variable(data, volatile=True)
        target = Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()
        
        pred = pred.cpu().numpy()
        target = target.data.cpu().numpy()
        all_pos += np.sum(target)
        true_pos += np.sum(pred[target.astype(bool)])

    test_loss = test_loss
    test_loss /= len(test_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:2.1f}%) Sensitivity {:2.2f}\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset),
            true_pos * 1.0 / all_pos
        ))

for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)

print("Finished")
