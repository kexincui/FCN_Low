import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
# from model import *
from fcn import *
from data_loading import *
from torchvision import transforms
import matplotlib.pyplot as plt

# parser = argparse.ArgumentParser(description="Satellite Classification")
# parser.add_argument("--batchSize", type=int, default=10, help="Training batch size")
# parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
# parser.add_argument("--milestone", type=int, default=5, help="When to decay learning rate; should be less than epochs")
# parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
# parser.add_argument("--outf", type=str, default="logs", help='path of log files')
# opt = parser.parse_args()

import easydict
args = easydict.EasyDict({
        "batchsize": 1,
        "epochs": 1,
        "gpu": 0,
        "milestone": 5,
        "lr": 1e-3,
        "outf": "logs",
})

# Load Dataset
# dataset_train = MultiDataSet("/home/ckx9411sx/deepGlobe/land-train")
dataset_train = MultiDataSet("/home/ckx9411sx/deepGlobe/resize")
dataset_test = MultiDataSet("/home/ckx9411sx/deepGlobe/land-train", '.jpg', True, transform = transforms.Resize((2448,2448)))

loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=args.batchsize, shuffle=True)
loader_test  = DataLoader(dataset=dataset_test,num_workers=4, batch_size = 1, shuffle = False)
print("# of training samples: %d\n" % int(len(dataset_train)))

# Build Model
model = UNet(7).cuda()
print(model)


# Move to GPU
if torch.cuda.is_available():
        model.cuda()
        
# Optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# Loss function
loss_func = nn.L1Loss()

# training
writer = SummaryWriter(args.outf)
step = 0

for epoch in range(args.epochs):
        if epoch < args.milestone:
            current_lr = args.lr
        else:
            current_lr = args.lr / 10.
        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)
        
        for i, data in enumerate(loader_train):
            # training step
            # print(i)
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            img_train = data
            satellite_img,class_img = Variable(img_train['satellite'].cuda()), Variable(img_train['class'].cuda())
            # print("****************************")
            out_train = model(satellite_img)
            loss = loss_func(out_train,class_img)
            loss.backward()
            optimizer.step()
            
            model.eval()
            out_train = torch.clamp(model(satellite_img), 1., 7.)
            print("[epoch %d][%d/%d] loss: %.4f" %
                (epoch+1, i+1, len(loader_train), loss.data[0]))
            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.data[0], step)
            step += 1
        ## the end of each epoch
model.eval()

for i, data in enumerate(loader_test):
    img_test = data
    satellite_img_test = Variable(img_test['satellite'].cuda())
    out_test = torch.clamp(model(satellite_img_test), 1., 7.)
    with open('test.txt', 'w') as outfile:
        outfile.write("%d\n" % out_test)
