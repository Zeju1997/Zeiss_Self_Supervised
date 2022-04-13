import os
from datasets.base_dataset import BaseDataset, get_transform
from PIL import Image
import random
import sys
import glob


class MnistDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir = os.path.join(opt.dir)  # create a path '/path/to/data/trainA'
        # self.paths = glob.glob(opt.dir+"*.png")  # generate a list of reference images
        self.paths = [os.path.basename(x) for x in glob.glob(opt.dir+"*.png")]  # generate a list of reference images

        # self.paths = open(os.path.join(opt.gta_list, 'gta5_' + opt.phase + '.txt')).readlines()

        self.size = len(self.paths)  # get the size of dataset A

        self.opt.input_nc = 3
        self.opt.output_nc = 3
        input_nc = self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.output_nc      # get the number of channels of output image

        self.transform = get_transform(self.opt)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """

        path = self.paths[index % self.size].strip('\n')  # make sure index is within then range

        # path = path[:-4] + ".jpg"

        path = os.path.join(self.dir, path)
        img = Image.open(path).convert('RGB')

        # apply image transformation
        transformed = self.transform(img)

        return transformed

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return self.size
