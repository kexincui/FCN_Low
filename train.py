import os
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize
from torchvision.transforms import ToTensor
import fcn
from data_loading import *
from criterion import *
from options import Options
import time
import visualizer

opt = Options().parse()
vis = visualizer.Visualizer(opt)
input_transform = Compose([
    ToTensor(),
    Normalize([.485, .456, .406], [.229, .224, .225]),
])


def load_network(opt, network, network_label, epoch_label):
    save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
    save_path = os.path.join(opt.checkpoints_dir, opt.name, save_filename)
    network.load_state_dict(torch.load(save_path))


def save_network(opt, network, network_label, epoch_label):
    save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
    save_path = os.path.join(opt.checkpoints_dir, opt.name, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda()

# Load Dataset
dataset_train = MultiDataSet(opt, transform = input_transform)

loader_train = DataLoader(dataset=dataset_train, num_workers=1, batch_size=opt.batchSize, shuffle=True)
dataset_size = int(len(dataset_train))
print("# of training samples: %d\n" % dataset_size)

# Build Model
model=fcn.FCN8(7).cuda()
print(model)


# Move to GPU
if torch.cuda.is_available():
        model.cuda()
        
# Optimizer
optimizer = optim.Adam(model.parameters(), lr=opt.lr)   #different with original version
# Loss function
criterion = CrossEntropyLoss2d()  # type: CrossEntropyLoss2d
criterion = criterion.cuda()
# training
model.train()

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0
    for i, data in enumerate(loader_train):
        # training step
        epoch_iter += opt.batchSize
        optimizer.zero_grad()
        img_train = data
        satellite_img,class_img = Variable(img_train['satellite'].cuda()), Variable(img_train['class'].cuda())
        # print("****************************")
        out_train = model.forward(satellite_img)
        loss = criterion(out_train, class_img)
        # print("out_train is {}".format(out_train[0].cpu().max(0)[1].data))
        # print("class_img[0].cpu().data is {}".format(class_img[0].cpu().data))
        loss.backward()
        optimizer.step()
        #visualization
        # vis.drawLine(torch.FloatTensor([epoch + i / dataset_size]), loss.cpu().data)
        # if epoch_iter % 1 == 0:
        #     vis.displayImg(inputImgTransBack(satellite_img), classToRGB(out_train[0].cpu().max(0)[1].data),
        #                    classToRGB(class_img[0].cpu().data))
        if epoch_iter % 10 == 0:
            print("Epoch {}, itr {}, loss is {}".format(epoch, i+1, loss.data[0]))
            print("Epoch {}, itr {}, loss is {}".format(epoch, i+1, loss.data[0]), file = open(opt.log_name, "a"))
    save_network(opt, model, "SateFCN", "latest")
    # print("Epoch {} model is saved as latest model".format(epoch))
    if epoch > opt.niter:
        current_lr = opt.lr - (opt.lr / opt.niter_decay) * (epoch - opt.niter)
    # set learning rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr
    if epoch % 10 == 0:
        save_network(opt, model, "SateFCN", epoch)
    print('learning rate {}'.format(current_lr), file = open(opt.log_name, "a"))
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time), file = open(opt.log_name, "a"))

print("!!!!!!!!!!!! Finish !!!!!!!!!!!!!!!!")
