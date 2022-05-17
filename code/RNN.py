import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch import nn
from tqdm import tqdm
import argparse
from loguru import logger
from utils.test_model import *

parser = argparse.ArgumentParser()

# Argument definition
parser.add_argument(
    'model',
    choices = ["LSTM", "GRU"],
    help='Model to train',
)
parser.add_argument(
    '--standardize',
    type=bool,
    default=True,
    help='Standardize the data or no',
)
parser.add_argument(
    '--components',
    type=int,
    default=10,
    help="Number of components for PCA"
)
parser.add_argument(
    '--ratio',
    type=float,
    default=0.2,
    help="Ratio of evaluation set"
)
parser.add_argument(
    '--workers',
    type=int,
    default=4,
    help="Number of workers for dataloader"
)
parser.add_argument(
    '--fc',
    type=int,
    default=2,
    help="Number of fully connected layers"
)
parser.add_argument(
    '--units',
    type=int,
    default=8,
    help="Number of hidden units"
)
parser.add_argument(
    '--layers',
    type=int,
    default=1,
    help="Number of hidden layers"
)
parser.add_argument(
    '--lr',
    type=float,
    default=1e-3,
    help="Learning rate"
)
parser.add_argument(
    '--epochs',
    type=int,
    default=50,
    help="Number of epochs to train the model"
)
parser.add_argument(
    '--dropout',
    type=float,
    default=0.3,
    help="Dropout for model"
)
parser.add_argument('--previous', dest='previous', action='store_true')
parser.add_argument('--no-previous', dest='previous', action='store_false')
parser.set_defaults(previous=True)
parser.add_argument(
    '--l2',
    type=float,
    default=1e-5,
    help="l2 penalty"
)
parser.add_argument(
    '--batchsize',
    type=int,
    default=15,
    help="batch size"
)
args = parser.parse_args()

logger.info(f"Params for the model: {args}")

data = np.load('../data/X_train_surge.npz')
data_y = pd.read_csv('../data/Y_train_surge.csv')

x_train = data["slp"]
data_y_1 = np.append(data["surge1_input"], data_y[surge1_columns].to_numpy(), axis=1)
data_y_2 = np.append(data["surge2_input"], data_y[surge2_columns].to_numpy(), axis=1)
y_train = np.stack([data_y_1, data_y_2], axis=2)

#creating the dataset
class SequenceDataset(Dataset):
    def __init__(self, X_train, Y_train, n_components=10, standardize=True, previous=False):
        """
        X_train: slp field of size [len, 40, 41, 41]
        Y_train: surge measurments of size [len, 20, 2]
        """
        l, t, w, h = X_train.shape
        assert w==h
        assert t==40
        self.standardize = standardize

        if standardize:
            self.scaler = StandardScaler()
            X_train= self.scaler.fit_transform(X_train.reshape(-1, 1)).reshape(l, t, w, h)

        self.n_components = n_components
        if n_components!=0:
            self._pca = IncrementalPCA(n_components=n_components)
            self.x = self._pca.fit_transform(X_train.reshape(l*t, w*h)).reshape(l, t, self.n_components)
        else:
            self.x = X_train.reshape(l, t, w*h)

        if previous:
            self.y = torch.tensor(Y_train).float()
        else:
            self.y = torch.tensor(Y_train[:, 10:, :]).float()

        self.x = torch.tensor(self.x).float()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i,:,:], self.y[i,:, :]

    def get_params(self):
        if self.standardize and self.n_components!=0:
            return self._pca.transform, self.scaler.transform
        elif not self.standardize and self.n_components!=0:
            return self._pca.transform, lambda a: a
        else:
            return lambda a: a, lambda b: b

class EvalSequenceDataset(SequenceDataset):
    def __init__(self, X_eval, Y_eval, previous, pca, scaler):
        l , t, w, h = X_eval.shape
        self.x = scaler(X_eval.reshape(-1, 1)).reshape(l, t, w, h)
        self.x = pca(X_eval.reshape(l*t, w*h)).reshape(l, t, -1)
        self.x = torch.tensor(self.x).float()
        if previous:
            self.y = torch.tensor(Y_eval).float()
        else:
            y=Y_eval[: ,10:, :]
            self.y = torch.tensor(y).float()

lim=int((1-args.ratio)*len(x_train))
indexes = np.arange(len(x_train))
np.random.shuffle(indexes)
train_indexes = indexes[:lim]
test_indexes = indexes[lim:]


logger.debug("Creating train dataset")
train_dataset = SequenceDataset(x_train[train_indexes], y_train[train_indexes], n_components=args.components, standardize=args.standardize, previous=args.previous)
logger.debug("Creating eval dataset")
eval_dataset = EvalSequenceDataset(x_train[test_indexes], y_train[test_indexes], args.previous, *train_dataset.get_params())

logger.info(f"Length of train dataset is {len(train_dataset)}")
logger.info(f"Length of eval dataset is {len(eval_dataset)}")

train_dataloader = DataLoader(  train_dataset, batch_size=args.batchsize, shuffle=True,
                                num_workers=args.workers, drop_last=False)
eval_dataloader = DataLoader(   eval_dataset, batch_size=args.batchsize, shuffle=True,
                                num_workers=args.workers, drop_last=False)


#Defining model
class ShallowRegressionLSTM(nn.Module):
    def __init__(self, components, hidden_units, num_layers, dropout=0.3, previous=False, fc=1):
        super().__init__()

        self.hidden_units = hidden_units
        self.num_layers = num_layers

        if components!=0:
            self.features = components  # this is the number of features
        else:
            self.features = 41*41
        self.lstm = nn.LSTM(
            input_size=self.features,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )

        if previous:
            self.out_shape = 2*20
        else:
            self.out_shape = 2*10

        probe_tensor = torch.zeros(1, 40, self.features)
        out, _ = self.lstm(probe_tensor)
        nb_features = out.view(1, -1).shape[1]

        modules = []
        if fc!=1:
            for i in range(fc-1):
                modules.append(nn.Dropout(dropout))
                modules.append(nn.ReLU())
                modules.append(nn.Linear(in_features=nb_features,out_features=nb_features))

        modules.append(nn.Dropout(dropout))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(in_features=nb_features,out_features=self.out_shape))

        self.linear = nn.Sequential(*modules)

    def forward(self, x):
        batch_size = x.shape[0]
        self.h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_().to(device)
        self.c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_().to(device)
        out, (hn, _) = self.lstm(x, (self.h0, self.c0))
        out = self.linear(out.reshape(batch_size, -1)).flatten()  # First dim of Hn is num_layers, which is set to 1 above.
        return out.view(-1, int(self.out_shape/2), 2)

class GRURNN(nn.Module):
    def __init__(self, components, hidden_units, num_layers, dropout=0.3, previous=False, fc=1):
        super().__init__()

        self.hidden_units = hidden_units
        self.num_layers = num_layers

        if components!=0:
            self.features = components  # this is the number of features
        else:
            self.features = 41*41
        self.gru = nn.GRU(
            input_size=self.features,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers,
            dropout=dropout,
        )

        probe_tensor = torch.zeros(1, 40, self.features)
        out, _ = self.gru(probe_tensor)
        nb_features = out.view(1, -1).shape[1]

        if previous:
            self.out_shape = 2*20
        else:
            self.out_shape=2*10

        modules = []
        if fc!=1:
            for i in range(fc-1):
                modules.append(nn.Dropout(dropout))
                modules.append(nn.ReLU())
                modules.append(nn.Linear(in_features=nb_features,out_features=nb_features))

        modules.append(nn.Dropout(dropout))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(in_features=nb_features,out_features=self.out_shape))

        self.linear = nn.Sequential(*modules)


    def forward(self, x):
        batch_size = x.shape[0]
        self.h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_().to(device)

        output, hn = self.gru(x,self.h0)
        out = self.linear(output.reshape(batch_size, -1)).flatten()
        return out.view(batch_size, int(self.out_shape/2), 2)

if args.model=="LSTM":
    model = ShallowRegressionLSTM(args.components, args.units, args.layers, dropout=args.dropout, previous=args.previous)
if args.model=="GRU":
    model = GRURNN(args.components, args.units, args.layers, dropout=args.dropout)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f"Total number of parameters: {pytorch_total_params}")

loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.l2)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

if args.previous:
    indexes = np.linspace(10, 19, 10, dtype=int)
else:
    indexes = [i for i in range(10)]

def train_model(data_loader, model, loss_function, optimizer):
    total_loss = 0
    model.train()
    n = 0
    y_real = pd.DataFrame(columns= surge1_columns +surge2_columns)
    y_pred = pd.DataFrame(columns= surge1_columns +surge2_columns)

    for X, y in data_loader:
        X, y = X.to(device), y.to(device)
        output = model(X)
        loss = loss_function(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        n += X.shape[0]
        total_loss += loss.item()*X.shape[0]

        surge1_pred = output[:, indexes, 0].detach().cpu().numpy()
        surge2_pred = output[:, indexes, 1].detach().cpu().numpy()

        surge1_real = y[:, indexes, 0].detach().cpu().numpy()
        surge2_real = y[:, indexes, 1].detach().cpu().numpy()
        y_pred = y_pred.append(pd.DataFrame(np.append(surge1_pred, surge2_pred, axis=1), columns=y_pred.columns))
        y_real = y_real.append(pd.DataFrame(np.append(surge1_real, surge2_real, axis=1), columns=y_real.columns))

    avg_loss = total_loss / n
    score = surge_prediction_metric(y_real, y_pred)
    logger.info(f"Train loss: {avg_loss}")
    logger.info(f"Score for train set: {score}")
    return avg_loss, score

def test_model(data_loader, model, loss_function):

    total_loss = 0
    n = 0
    model.eval()
    y_real = pd.DataFrame(columns= surge1_columns +surge2_columns)
    y_pred = pd.DataFrame(columns= surge1_columns +surge2_columns)

    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            total_loss += loss_function(output, y).item()*X.shape[0]
            n += X.shape[0]

            surge1_pred = output[:, indexes, 0].cpu().numpy()
            surge2_pred = output[:, indexes, 1].cpu().numpy()

            surge1_real = y[:, indexes, 0].cpu().numpy()
            surge2_real = y[:, indexes, 1].cpu().numpy()
            y_pred = y_pred.append(pd.DataFrame(np.append(surge1_pred, surge2_pred, axis=1), columns=y_pred.columns))
            y_real = y_real.append(pd.DataFrame(np.append(surge1_real, surge2_real, axis=1), columns=y_real.columns))

    avg_loss = total_loss / n
    score = surge_prediction_metric(y_real, y_pred)
    logger.info(f"Test loss: {avg_loss}")
    logger.info(f"Score for test set: {score}")
    return avg_loss, score


logger.info("Untrained test\n--------")
test_model(eval_dataloader, model, loss_function)
print()
test_scores = []
train_scores = []
train_losses = []
test_losses = []

for ix_epoch in range(args.epochs):
    logger.info(f"Epoch {ix_epoch}\n---------")
    train_loss, train_score = train_model(train_dataloader, model, loss_function, optimizer=optimizer)
    test_loss, test_score = test_model(eval_dataloader, model, loss_function)
    scheduler.step(test_loss)

    test_scores.append(test_score)
    train_scores.append(train_score)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    print()

fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(test_scores)
ax1.grid()
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Score")
ax1.set_title("Test score")

ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(test_losses)
ax2.grid()
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Loss")
ax2.set_title("Test loss")

ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(train_scores)
ax3.grid()
ax3.set_xlabel("Epochs")
ax3.set_ylabel("Score")
ax3.set_title("Train score")


ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(train_losses)
ax4.grid()
ax4.set_xlabel("Epochs")
ax4.set_ylabel("Loss")
ax4.set_title("Train loss")

plt.show()
