import argparse
import os
import utils
import torch
import sys


class Options(object):
    def __init__(self):
        print(sys.argv)
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--dataroot', required=True,
                                 help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--loadSizeX', type=int, default=1224, help='scale images to this size')
        self.parser.add_argument('--loadSizeY', type=int, default=1224, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=512, help='then crop to this size')
        self.parser.add_argument('--name', type=str, default='experiment_name',
                                 help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--save_epoch_freq', type=int, default=10,
                                 help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--continue_train', action='store_true',
                                 help='continue training: load the latest model')
        self.parser.add_argument('--epoch_count', type=int, default=1,
                                 help='the starting epoch count, we save the model by <epoch_count>, '
                                      '<epoch_count>+<save_latest_freq>, ...')
        self.parser.add_argument('--which_epoch', type=str, default='latest',
                                 help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--niter', type=int, default=300, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=300,
                                 help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
        self.opt = self.parser.parse_args()

    def parse(self):

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        utils.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        self.opt.log_name = os.path.join(expr_dir, 'loss.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
