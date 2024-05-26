import os
from dataclasses import dataclass
import torch.nn as nn
import torch.nn.functional as F
from tkinter import filedialog

@dataclass
class ModelArgs:
    filename: str = 'C:\\Users\\meyer\\Documents\\Projects\\travnet\\models\\travnetCNN16k.pth'
    num_channels: int = 1
    sample_rate: int = 30000
    threshold: int = -9

class Findweights:
    def __init__(self, model_args: ModelArgs):        
        if model_args.filename is not None:
            self.filename = model_args.filename            
            print(f'Filename: {ModelArgs.filename}')
        else:
            print('Please select a file to import. Tip: use the window, which sometimes appears beind the current windows on the desktop.')
            current_directory = os.getcwd()
            full_path = filedialog.askopenfilename(initialdir=current_directory)
            ModelArgs.path, ModelArgs.filename = os.path.split(full_path)

            print("Directory Path:", ModelArgs.path)  # prints the directory path
            print("Filename:", ModelArgs.filename)  # prints the filename            
            self.filename = full_path            
        return None    

class CNN(nn.Module):    
    def __init__(self):
        super(CNN,self).__init__()
        self.cnn_model = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3),
        nn.BatchNorm2d(6),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=6, out_channels=32, kernel_size=2),        
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2))
        
        self.fc_model = nn.Sequential(
        nn.Linear(in_features=1568, out_features=5000),        
        nn.BatchNorm1d(5000),
        nn.ReLU(),
        nn.Linear(in_features=5000, out_features=1000),        
        nn.BatchNorm1d(1000),
        nn.ReLU(),
        nn.Linear(in_features=1000, out_features=6))
        
    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1)
        x = self.fc_model(x)
        x = F.softmax(x, dim=1)
        
        return x
    