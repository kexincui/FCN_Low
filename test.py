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


def load_network(opt, network, network_label, epoch_label):
    save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
    save_path = os.path.join(opt.checkpoints_dir, opt.name, save_filename)
    network.load_state_dict(torch.load(save_path))


opt = Options().parse()
opt.train = False
vis = visualizer.Visualizer(opt)
input_transform = Compose([
    ToTensor(),
    Normalize([.485, .456, .406], [.229, .224, .225]),
])

# Load Dataset
dataset_test = MultiDataSet(opt, transform=input_transform)

loader_test = DataLoader(dataset=dataset_test, num_workers=1, batch_size=opt.batchSize, shuffle=True)
dataset_size = int(len(loader_test))
print("# of testing samples: %d\n" % dataset_size)

# Build Model
model = fcn.FCN8(7).cuda()
print(model)

# Move to GPU
if torch.cuda.is_available():
    model.cuda()


# testing
model.eval()
load_network(opt, model, "SateFCN", opt.which_epoch)

for epoch in range(10):
    epoch_start_time = time.time()
    epoch_iter = 0
    for i, data in enumerate(loader_test):
        # testing step
        epoch_iter += opt.batchSize
        img_test = data
        satellite_img, class_img = Variable(img_test['satellite'].cuda(), volatile = True), Variable(img_test['class'].cuda(), volatile = True)
        # print("****************************")
        out_test = model.forward(satellite_img)
        print("display img{}".format(i))
        # visualization
        vis.displayImg(inputImgTransBack(satellite_img), classToRGB(out_test[0].cpu().max(0)[1].data),
                       classToRGB(class_img[0].cpu().data))


print("!!!!!!!!!!!! Finish !!!!!!!!!!!!!!!!")
