#!/usr/bin/python3

import sys
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
from additional_data import *

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
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=80, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--feats', type=bool, default=True,
                    help='use additional data features')
parser.add_argument('--weight', type=float, default=25.0,
                    help='Additional weight factor for positive events')
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
df.reset_index(drop=True, inplace=True)

Y = df['contact'].as_matrix() * 1
categorical_columns = ['country', 'region', 'metro', 'devicetype', 'osfamily', 'browser', 'devicemake']

additional_dfs = load_all_csv()
additional_cols = [
        ('browser.csv', 'browser'),
        ('city.csv', 'city'),
        ('device.csv', 'devicetype'),
        ('ip.csv', 'ipaddress'),
        ('metro.csv', 'metro'),
        ('os.csv', 'osfamily'),
        ('region.csv', 'region'),
        ('zip.csv', 'zip'),
]

df['dow'] = df['logentrytime'].dt.dayofweek
df['hour'] = df['logentrytime'].dt.hour
df['month'] = df['logentrytime'].dt.month
df['day'] = df['logentrytime'].dt.day
categorical_columns += ['dow', 'hour', 'month', 'day']

XCat = np.zeros([df.shape[0], len(categorical_columns)], dtype=np.int64)
col_unique = {c: np.unique(df[c].as_matrix()).shape[0] for c in categorical_columns}
label_encoders = [preprocessing.LabelEncoder() for c in categorical_columns]
for i in range(len(categorical_columns)):
    l = label_encoders[i]
    c = categorical_columns[i]
    XCat[:,i] = l.fit_transform(df[c].tolist())

X = np.array(df.index.tolist())
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
        if args.feats:
            sizes[0] += sum([additional_dfs[a[0]][1] for a in additional_cols])
        self.layers = nn.ModuleList([nn.Linear(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)])

    def forward(self, categories, feats):
        x = [self.embeddings[i](torch.unsqueeze(categories[:,i], 0))[0] for i in range(len(categorical_columns))]
        if args.feats:
            x += [feats]
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

def get_batch(data, volatile=False):
    data = data.numpy()
    categories = XCat[data]
    df_slice = df.iloc[data]
    feats = [inner_join(df_slice, additional_dfs[fn], original_column_name) for fn, original_column_name in additional_cols]
    feats = np.concatenate(feats, axis=1).astype(np.float32)
    res = categories, feats
    res = [torch.from_numpy(v) for v in res]
    if args.cuda:
        res = [v.cuda() for v in res]
    res = [Variable(v, volatile=volatile) for v in res]
    return tuple(res)

def train(epoch):
    model.train()
    train_weights = torch.from_numpy(np.array([1.0,args.weight])).float().cuda()
    criterion = torch.nn.NLLLoss(train_weights)
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            target = target.cuda()
        target = Variable(target)
        categories, feats = get_batch(data)
        optimizer.zero_grad()
        output = model(categories, feats)
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
            target = target.cuda()
        target = Variable(target)
        categories, feats = get_batch(data, volatile=True)
        output = model(categories, feats)
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

# Print Embedding Plots
for i, c in enumerate(categorical_columns):
    V = model.embeddings[i].weight.data.cpu().numpy()
    if embedding_sizes[c] > 2:
        V = PCA(n_components=2).fit_transform(V)
    X = V[:,0]
    Y = V[:,1]
    labels = np.array(label_encoders[i].inverse_transform(range(col_unique[c])))
    if labels.size > 20:
        freqs = Counter(df[c])
        dispLabels = {f[0] for f in freqs.most_common(20)}
        disp = np.array([(l in dispLabels) for l in labels])
        X = X[disp]
        Y = Y[disp]
        labels = labels[disp]
    plt.cla()
    plt.clf()
    fig, ax = plt.subplots()
    ax.scatter(X,Y)
    for l, x, y in zip(labels, X, Y):
        ax.annotate(l, (x,y))
    #plt.show()
    plt.title('%s Embedding' % c)
    plt.savefig('outputs/embed_%s.png' % c)

print("Finished")
