import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

def train(dataloader_train: DataLoader,
          dataloader_test: DataLoader,
          device: str,
          model: nn.Module,
          epochs: int,
          learning_rate: float,
          save: bool):

    optimiser = optim.Adam(model.parameters(),lr=learning_rate)
    history = []

    epoch_bar = trange(epochs)
    for epoch in epoch_bar:

        # train
        model.train()
        for batch,data in enumerate(dataloader_train):
            x,y = data
            x,y = x.to(device),(y.view(-1)).to(device)

            optimiser.zero_grad()

            out = model(x)
            loss = F.cross_entropy(out,y)

            loss.backward()
            optimiser.step()

        # test
        running_loss = 0
        running_acc  = 0
        for batch,data in enumerate(dataloader_test):
            x,y = data
            x,y = x.to(device),(y.view(-1)).to(device)
            outs = model(x)

            test_acc = ((torch.argmax(outs,1))== y).cpu().detach().numpy().sum()/len(y)*100
            test_loss = F.cross_entropy(outs,y).item()
            running_acc  += test_acc*x.size(0)
            running_loss += test_loss*x.size(0)

        test_size = len(dataloader_test.dataset)
        test_acc = running_acc/test_size
        test_loss = running_loss/test_size
        epoch_bar.set_description('acc={0:.2f}%\tcross entropy={1:.4f}'
                                  .format(test_acc, test_loss))

        history.append((test_acc,test_loss))

    if save:
        #save
        pass

    return model,history

