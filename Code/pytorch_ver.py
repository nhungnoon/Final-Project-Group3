# code reference
# https://stackoverflow.com/questions/55588201/pytorch-transforms-on-tensordataset
# https://github.com/amir-jafari/Deep-Learning/blob/master/Pytorch_/CNN/1_ImageClassification/example_MNIST.py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, TensorDataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import cv2

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, (3, 3))
        self.convnorm1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d((2, 2))

        self.conv2 = nn.Conv2d(16, 32, (3, 3))
        self.convnorm2 = nn.BatchNorm2d(32)
        self.pool2 = nn.AvgPool2d((2, 2))

        self.linear1 = nn.Linear(32 * 5 * 5, 400)
        self.linear1_bn = nn.BatchNorm1d(400)
        self.drop = nn.Dropout(0.5)
        self.linear2 = nn.Linear(400, 25)
        self.act = torch.relu

    def forward(self, x):
        x = self.pool1(self.convnorm1(self.act(self.conv1(x))))
        x = self.pool2(self.convnorm2(self.act(self.conv2(x))))
        x = self.drop(self.linear1_bn(self.act(self.linear1(x.view(len(x), -1)))))
        return self.linear2(x)


# load the data
train = pd.read_csv('sign_mnist_train.csv')
test = pd.read_csv('sign_mnist_test.csv')


# create images from the values
train_imgs = train.iloc[:,1:].values.reshape((train.iloc[:,1:].shape[0],28,28))
train_imgs = np.expand_dims(train_imgs, axis=-1)
# get the labels from the first column
train_labels = train.iloc[:, :1].values

#print(len(np.unique(train_labels)))

# create images from the values
test_imgs = test.iloc[:,1:].values.reshape((test.iloc[:,1:].shape[0],28,28))
test_imgs = np.expand_dims(test_imgs, axis=-1)


# get the labels from the first column
test_labels = test.iloc[:, :1].values

#x_train, x_test2, y_train, y_test2 = train_test_split(train_imgs, train_labels, test_size=0.30, random_state=0)
x_train, x_test2, y_train, y_test2 = train_test_split(train_imgs, train_labels, stratify=train_labels, test_size=0.30, random_state=0)


x_train1 = Variable(torch.FloatTensor(x_train), requires_grad = True).view(len(x_train), 1,28,28)
y_train1 = Variable(torch.LongTensor(y_train), requires_grad=False)

x_test2= Variable(torch.FloatTensor(x_test2), requires_grad = False).view(len(x_test2), 1,28,28)
y_test2 = Variable(torch.LongTensor(y_test2), requires_grad=False)


x_test = Variable(torch.FloatTensor(test_imgs), requires_grad = False).view(len(test_imgs), 1,28,28)
y_test = Variable(torch.LongTensor(test_labels), requires_grad=False)

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)

def hv_flip(tensor):
    """Flips tensor horizontally or vertically"""

    if np.random.rand() <0.3:
        tensor = tensor.flip(2)
    if np.random.rand() > 0.5:
        tensor = tensor.flip(1)
    return tensor



def test_transform(tensor):
    all_tranform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomAffine(20, translate=(0.2, 0.4), shear = (0.1, 0.3)),
                transforms.ToTensor()
            ])

    tensor = all_tranform(tensor)
    return tensor

data_train = CustomTensorDataset(tensors=(x_train1, y_train1), transform=hv_flip)

train_loader = torch.utils.data.DataLoader(data_train, batch_size=400, shuffle = True)

# %% ----------------------------------- Helper Functions --------------------------------------------------------------
def acc(x, y, return_labels=False):
    with torch.no_grad():
        logits = model(x)
        _, pred_labels = torch.max(logits.data, 1)
    if return_labels:
        return pred_labels
    else:
        return 100*accuracy_score(y, pred_labels)



# %% -------------------------------------- Training Prep ----------------------------------------------------------
model = CNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
criterion = nn.CrossEntropyLoss()

# %% -------------------------------------- Training Loop ----------------------------------------------------------
N_EPOCHS = 20
for epoch in range(N_EPOCHS):
    losses = []
    model.train()
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()
        logits = model(inputs)
        labels = labels.squeeze_()
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.data.mean())
    print('[%d/%d] Train Loss: %.3f' % (epoch + 1, N_EPOCHS, np.mean(losses)))
    model.eval()
    with torch.no_grad():
        x_test_split = Variable(x_test2)
        y_test_pred_split = model(x_test_split)
        y_test_split = y_test2.squeeze_()
        loss = criterion(y_test_pred_split, y_test2)
        loss_test_split = loss.item()

        x_test_ori = Variable(x_test)
        y_test_pred_ori = model(x_test_ori)
        y_test_ori = y_test.squeeze_()
        loss_ori = criterion(y_test_pred_ori, y_test_ori)
        loss_test_ori = loss_ori.item()


    print("Epoch {} | Valid Loss: {:.5f}, Valid Acc: {:.2f}, Test Loss: {:.5f}, Test Acc: {:.2f}".format(
        epoch, loss_test_split, acc(x_test2, y_test2), loss_test_ori, acc(x_test, y_test)))

# Testing


def predict(x):
    images = []
    for img_path in x:
        # read in images in grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # Resize the image to be a rectangle: increase the size from previous version
        img = cv2.resize(img, (28, 28))
        # append to the images
        images.append(img)
    x = Variable(torch.FloatTensor(images), requires_grad = True).view(len(images), 1,28,28)

    model.eval()
    test_output = model(x)
    _, predicted_test = torch.max(test_output.data, 1)
    return predicted_test

self_create_imgs = ['/home/ubuntu/gpu_noon/letter_h.jpeg', '/home/ubuntu/gpu_noon/letter_a.jpeg', '/home/ubuntu/gpu_noon/letter_p.jpeg',
'/home/ubuntu/gpu_noon/letter_p.jpeg','/home/ubuntu/gpu_noon/letter_y.jpeg', '/home/ubuntu/gpu_noon/letter_h.jpeg',
'/home/ubuntu/gpu_noon/letter_o.jpeg', '/home/ubuntu/gpu_noon/letter_l.jpeg', '/home/ubuntu/gpu_noon/letter_l.jpeg',
'/home/ubuntu/gpu_noon/letter_i.jpeg','/home/ubuntu/gpu_noon/letter_d.jpeg', '/home/ubuntu/gpu_noon/letter_a.jpeg',
'/home/ubuntu/gpu_noon/letter_y.jpeg']
self_pred = predict(self_create_imgs)
print(self_pred)
