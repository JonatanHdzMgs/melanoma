from urllib.request import urlopen
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
import os
import timm
import cv2
import pandas as pd

#code to visualize
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm


def transform(images):
    return (torch.tensor((cv2.resize(images, (224,224))))).permute(2, 0, 1)
 
class CustomDataSet(torch.utils.data.Dataset):
    def __init__(self, dataframe, transform, stage, path):
        if stage == 'test':
            dataframe = dataframe.iloc[:7000]
        elif stage == 'train':
            dataframe = dataframe.iloc[7000:]
        self.dataframe = dataframe
        self.transform = transform
        self.path = path
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        image_path = self.dataframe.iloc[index, 1]
        print(image_path)
        print(os.path.join(self.path, image_path))
        image = cv2.imread(os.path.join(self.path, image_path))
        
        image = image.astype(np.float32) / 255.0
        label = self.dataframe.iloc[index, 3]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def training_function(dataloader, model, optimizer):
    model.train()
    loss_total = 0
    for batch in tqdm(dataloader):
        images, targets = batch
        output = model(images)
        loss = nn.functional.cross_entropy(output, targets) 
        loss.backward()
        optimizer.step()
        loss_total = loss_total + loss
    return loss_total/(len(dataloader))

def validation_function(dataloader, model):
    model.eval()
    loss_total = 0
    for batch in dataloader:
        images, targets = batch
        output = model(images)
        loss = nn.functional.cross_entropy(output, targets) 
        loss_total = loss_total + loss
    return loss_total/(len(dataloader))

dataframe = pd.read_csv('/Users/lucasbautista/Documents/images_to_class_melanoma.csv')
#this is the csv file with image names and 1s and 0s
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
test_dataset_root = "/Users/lucasbautista/Downloads/train"
training_dataset_root = '/Users/lucasbautista/Downloads/train'

batch_size = 8 
training_dataset = CustomDataSet(dataframe, transform, 'train', test_dataset_root)
test_dataset = CustomDataSet(dataframe, transform, 'test', test_dataset_root)
training_dataloader = DataLoader(training_dataset, batch_size = batch_size, shuffle = True)
test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

model = timm.create_model('resnet18.a1_in1k', pretrained=True)
model = model.train()

#dummy_image = cv2.imread('/Users/lucasbautista/Documents/6.1010/image_processing/bluegill_inverted.png')
#dummy_image = dummy_image.astype(np.float32) / 255.0
#print(dummy_image.shape)
#output = model((torch.tensor((cv2.resize(dummy_image, (224,224))))).unsqueeze(0).permute(0, 3, 1, 2))
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

training_function(training_dataloader, model, optimizer)
torch.save(model, '/Users/lucasbautista/Documents/parameters.pt')

