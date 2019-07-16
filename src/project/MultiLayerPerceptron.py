import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLayerPerceptron(nn.Module):
    def __init__(self,n_in,n_classes):
        super(MultiLayerPerceptron,self).__init__()
        self.n_in  = n_in
        self.n_classes = n_classes
        
        self.fc1   = nn.Linear( n_in,500)
        self.fc2   = nn.Linear(500,500)
        self.fc3   = nn.Linear(500,  self.n_classes)

        
    def forward(self, x: torch.Tensor):
        x = F.dropout(x,p=0.1)

        x = F.relu(self.fc1(x))
        x = F.dropout(x,p=0.2)
        
        x = F.relu(self.fc2(x))
        x = F.dropout(x,p=0.2)
        
        x = F.relu(self.fc3(x))
        x = F.dropout(x,p=0.3)
        
        x = x.view(-1,self.n_classes)
        
        return F.log_softmax(x,1)
