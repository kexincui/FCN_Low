#Prepare Dataset
from os import listdir
from os.path import join
import torch
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageFile
import random
from torchvision.transforms import ToTensor
import cv2

ImageFile.LOAD_TRUNCATED_IMAGES = True


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def find_label_map_name(img_filenames,labelExtension = ".png"):
    img_filenames = img_filenames.replace('_sat.jpg','_mask')
    return img_filenames + labelExtension


def RGB_mapping_to_class(label):
    label=label.transpose((1,2,0))
    l,w = label.shape[0],label.shape[1]
    classmap = np.zeros(shape=(l,w))
    indices = np.where(np.all(label == (0,255,255), axis=-1))
    classmap[indices[0].tolist(),indices[1].tolist()]=1
    indices = np.where(np.all(label == (255,255,0), axis=-1))
    classmap[indices[0].tolist(),indices[1].tolist()]=2
    indices = np.where(np.all(label == (255,0,255), axis=-1))
    classmap[indices[0].tolist(),indices[1].tolist()]=3
    indices = np.where(np.all(label == (0,255,0), axis=-1))
    classmap[indices[0].tolist(),indices[1].tolist()]=4
    indices = np.where(np.all(label == (0,0,255), axis=-1))
    classmap[indices[0].tolist(),indices[1].tolist()]=5
    indices = np.where(np.all(label == (255,255,255), axis=-1))
    classmap[indices[0].tolist(),indices[1].tolist()]=6
    indices = np.where(np.all(label == (0,0,0), axis=-1))
    classmap[indices[0].tolist(),indices[1].tolist()]=0
#     plt.imshow(colmap)
#     plt.show()
    return classmap


def classToRGB(label):
    l,w = label.shape[0],label.shape[1]
    colmap = np.zeros(shape=(l,w,3))
    indices = np.where(label == 1)
    colmap[indices[0].tolist(),indices[1].tolist(),:]=[0,255,255]
    indices = np.where(label == 2)
    colmap[indices[0].tolist(),indices[1].tolist(),:]=[255,255,0]
    indices = np.where(label == 3)
    colmap[indices[0].tolist(),indices[1].tolist(),:]=[255,0,255]
    indices = np.where(label == 4)
    colmap[indices[0].tolist(),indices[1].tolist(),:]=[0,255,0]
    indices = np.where(label == 5)
    colmap[indices[0].tolist(),indices[1].tolist(),:]=[0,0,255]
    indices = np.where(label == 6)
    colmap[indices[0].tolist(),indices[1].tolist(),:]=[255,255,255]
    indices = np.where(label == 0)
    colmap[indices[0].tolist(),indices[1].tolist(),:]=[0,0,0]
    transform = ToTensor();
#     plt.imshow(colmap)
#     plt.show()
    return transform(colmap.astype(np.uint8))


def inputImgTransBack(inputs):
    image = inputs[0].cpu().data
    image[0] = image[0] * .229 + .485
    image[1] = image[1] * .224 + .456
    image[2] = image[2] * .225 + .406
    return image


class MultiDataSet(data.Dataset):
    """input and label image dataset"""

    def __init__(self, opt,labelExtension = '.png', testFlag = False,transform = None):
        super(MultiDataSet,self).__init__()
        """
        Args:

        fileDir(string):  directory with all the input images.
        transform(callable, optional): Optional transform to be applied on a sample
        """
        self.opt = opt
        self.fileDir = join(opt.dataroot, "train" if opt.train == True else "test")
        self.labelExtension = labelExtension
        self.testFlag = testFlag
        self.image_filenames  = [image_name for image_name in listdir(self.fileDir+'/Sat') if is_image_file(image_name)]
        self.transform = transform
        self.classdict = {1:"urban",2:"agriculture",3:"rangeland",4:"forest",5:"water",6:"barren",0:"unknown"}
   
    def __getitem__(self, index):
        Satsample = Image.open(join(self.fileDir,'Sat/'+self.image_filenames[index]))
        Satsample = Satsample.resize((self.opt.loadSizeX, self.opt.loadSizeY), Image.NEAREST)
        if self.testFlag == False:
            labelsamplename= find_label_map_name(self.image_filenames[index],self.labelExtension)
            labelsample = Image.open(join(self.fileDir,'Label/'+labelsamplename))
            labelsample = labelsample.resize((self.opt.loadSizeX, self.opt.loadSizeY), Image.NEAREST)
        if self.transform:
            Satsample = self.transform(Satsample)
        l,w=Satsample.shape[1],Satsample.shape[2]
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        l_offset = random.randint(0, max(0, l - self.opt.fineSize - 1))

        Satsample = Satsample[:, l_offset:l_offset + self.opt.fineSize,
            w_offset:w_offset + self.opt.fineSize]

        classmap=np.zeros(shape=(l,w))
        if self.testFlag == False:                                  
            labelsample = np.array(labelsample).transpose((2,0,1))
            labelsample = labelsample[:, l_offset:l_offset + self.opt.fineSize,
            w_offset:w_offset + self.opt.fineSize]
            classmap = RGB_mapping_to_class(labelsample)
        sample = {'satellite':torch.Tensor(Satsample),'class':torch.LongTensor(classmap)} 
        # colmap = classToRGB(label)
        # plt.imshow(colmap)
        # plt.show() 
        return sample

    def __len__(self):
        return len(self.image_filenames)

# dataset_train = MultiDataSet("/home/ckx9411sx/deepGlobe/land-train")
# np.save('/home/ckx9411sx/deepGlobe/temp/train_data_mc.npy', dataset_train)
# print(dataset_train)