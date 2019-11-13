import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image

'''
copied from https://github.com/WonwoongCho/Generative-Inpainting-pytorch/blob/daef8659cb0e15359f32a63c2159f17d75a555e6/data_loader.py
'''
class CelebDataset(Dataset):
    def __init__(self, transform, mode, FLAGS):
        
        self.transform = transform
        self.mode = mode
        self.FLAGS = FLAGS
        #self.lines = open(metadata_path, 'r').readlines()
        #self.num_data = int(self.lines[0])

        print ('Start preprocessing dataset..!')
        #random.seed(1234)
        self.preprocess()
        print ('Finished preprocessing dataset..!')

        if self.mode == 'train':
            self.num_data = len(self.train_filenames)
        elif self.mode == 'test':
            self.num_data = len(self.test_filenames)

    def preprocess(self):

        # 000001.png - 100000.png to train,
        # 100001.png - 101000.png to test
        self.train_filenames = ['%06d.png'%i for i in range(1,100001)]
        self.test_filenames = ['%06d.png'%i for i in range(100001,101001)]

    def __getitem__(self, index):
        if self.mode == 'train':
            image = Image.open(os.path.join(self.FLAGS.image_path, self.train_filenames[index]))
            if self.FLAGS.guided:
                edge = Image.open(os.path.join(self.FLAGS.edge_path, self.train_filenames[index]))
        elif self.mode in ['test']:
            image = Image.open(os.path.join(self.FLAGS.image_path, self.test_filenames[index]))
            if self.FLAGS.guided:
                edge = Image.open(os.path.join(self.FLAGS.edge_path, self.test_filenames[index]))
        # self.check_size(image, index)
        if self.FLAGS.guided:
            return self.transform(image),self.transform(edge)
        else:
            return self.transform(image)

    def __len__(self):
        return self.num_data


def get_loader(batch_size,FLAGS,dataset='CelebA', mode='train'):
    """Build and return data loader."""

    if mode == 'train':
        transform = transforms.Compose([
            transforms.CenterCrop(FLAGS.crop_size),
            transforms.Resize(FLAGS.img_size, interpolation=Image.ANTIALIAS),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    else:
        transform = transforms.Compose([
            transforms.CenterCrop(FLAGS.crop_size),
            transforms.Scale(FLAGS.img_size, interpolation=Image.ANTIALIAS),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    if dataset == 'CelebA':
        dataset = CelebDataset(transform, mode,FLAGS)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle)
    return data_loader