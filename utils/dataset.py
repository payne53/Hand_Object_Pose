# -*- coding: utf-8 -*-

# import libraries
import numpy as np
import os
import torch.utils.data as data
from PIL import Image


"""# Load Dataset"""

class Dataset(data.Dataset):

    def __init__(self, root='./', load_set='train', transform=None, no_image=False):
        self.root = root#os.path.expanduser(root)
        self.transform = transform
        self.load_set = load_set  # 'train','val','test'
        self.no_image = no_image

        self.images = np.load(os.path.join(root, 'images-%s.npy'%self.load_set))
        self.points2d = np.load(os.path.join(root, 'points2d-%s.npy'%self.load_set))
        # self.points2d = np.load(os.path.join(root, 'points2d-estimate-%s.npy'%self.load_set))
        self.points3d = np.load(os.path.join(root, 'points3d-%s.npy'%self.load_set))
        
        #if shuffle:
        #    random.shuffle(data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, points2D, points3D).
        """
        if self.no_image:
            point2d = self.points2d[index]
            point3d = self.points3d[index]
            return point2d, point3d
        else:
            point2d = self.points2d[index]
            point3d = self.points3d[index]
            image = Image.open(self.images[index])
            if self.transform is not None:
                image = self.transform(image)
            return image[:3], point2d, point3d

    def __len__(self):
        return len(self.images)
