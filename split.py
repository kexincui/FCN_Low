#Prepare Dataset
from os import listdir
from os.path import join
from PIL import Image
import numpy as np


def split2list(images, split, default_split=0.9):
    if isinstance(split, str):
        with open(join(split, "train_list")) as f:
            train_samples = [x.strip() for x in f.readlines()]
        with open(join(split, "test_list")) as f:
            test_samples = [x.strip() for x in f.readlines()]
        return train_samples, test_samples
    elif isinstance(split, float):
        split_values = np.random.uniform(0,1,len(images)) < split
    else:
        split_values = np.random.uniform(0,1,len(images)) < default_split
    train_samples = [sample for sample, split in zip(images, split_values) if split]
    test_samples = [sample for sample, split in zip(images, split_values) if not split]
    return train_samples, test_samples


def find_label_map_name(img_filenames,labelExtension = ".png"):
    img_filenames = img_filenames.replace('_sat.jpg','_mask')
    return img_filenames + labelExtension


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


fileDir  = "/scratch/user/jiangziyu/data/"
image_filenames  = [image_name for image_name in listdir(fileDir+'/Sat') if is_image_file(image_name)]
train_samples, test_samples = split2list(image_filenames, 0.75)
for train_name in train_samples:
    print(train_name, file = open("train_list", "a"))
for test_name in test_samples:
    print(test_name, file = open("test_list", "a"))


for i, train_name in enumerate(train_samples):
    Satsample = Image.open(join(fileDir, 'Sat/' + train_name))
    labelsamplename = find_label_map_name(train_name, ".png")
    labelsample = Image.open(join(fileDir, 'Label/' + labelsamplename))
    Satsample.save(join(fileDir, 'train/Sat/' + train_name))
    labelsample.save(join(fileDir, 'train/Label/' + labelsamplename))


for i, test_name in enumerate(test_samples):
    Satsample = Image.open(join(fileDir, 'Sat/' + test_name))
    labelsamplename = find_label_map_name(test_name, ".png")
    labelsample = Image.open(join(fileDir, 'Label/' + labelsamplename))
    Satsample.save(join(fileDir, 'test/Sat/' + test_name))
    labelsample.save(join(fileDir, 'test/Label/' + labelsamplename))