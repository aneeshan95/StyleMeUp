import torch
import torchvision.utils as tutils
import torch.utils.data as data
import torchvision.transforms as transforms
import os
from random import randint
from PIL import Image
import random
import torchvision.transforms.functional as F
import numpy as np
import argparse


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FGSBIR_Dataset(data.Dataset):
    def __init__(self, hp, mode):

        self.hp = hp
        self.mode = mode

        self.photo_root = '/vol/research/sketchCV/datasets/sketchy_rendered_256x256/photo/tx_000000000000/'
        self.sketch_root = '/vol/research/sketchCV/datasets/sketchy_rendered_256x256/sketch/tx_000000000000/'

        self.all_classes = os.listdir(self.sketch_root)
        self.all_classes.sort()

        sketch_samples = []
        # getting all_samples
        for class_name in self.all_classes:
            sketch_samples.extend([(class_name+'/'+x[:-4]) for x in os.listdir(self.sketch_root+class_name)])

        self.Train_Sketch, self.Test_Sketch = [], []
        self.class_dict_Train, self.class_dict_Test = {}, {}
        for class_name in self.all_classes:
            per_class_data = np.array([x for x in sketch_samples if class_name == x.split('/')[0]])
            per_class_Train = per_class_data[random.sample(range(len(per_class_data)),
                                                           int(len(per_class_data) * hp.splitTrain))]
            per_class_Test = set(per_class_data) - set(per_class_Train)
            self.Train_Sketch.extend(list(per_class_Train))
            self.Test_Sketch.extend(list(per_class_Test))

        print('Total Training Sample {}'.format(len(self.Train_Sketch)))
        print('Total Testing Sample {}'.format(len(self.Test_Sketch)))


        self.train_transform = get_ransform('Train')
        self.test_transform = get_ransform('Test')

    def __getitem__(self, item):

        sample = {}
        if self.mode == 'Train':
            sketch_path = self.Train_Sketch[item]
            # sketch_path example :
            positive_sample = sketch_path.split('-')[0]

            possible_list = list(range(len(self.all_classes)))
            possible_list.remove(self.all_classes.index(positive_sample.split('/')[0]))
            negative_class = self.all_classes[possible_list[randint(0, len(possible_list) - 1)]]
            negative_items = os.listdir(self.photo_root+negative_class)
            negative_sample = negative_class + '/' + negative_items[randint(0, len(negative_items) - 1)][:-4]


            sketch_img = Image.open(self.sketch_root + sketch_path + '.png').convert('RGB')
            positive_img = Image.open(self.photo_root + positive_sample + '.jpg').convert('RGB')
            negative_img = Image.open(self.photo_root + negative_sample + '.jpg').convert('RGB')

            n_flip = random.random()
            if n_flip > 0.5:
                sketch_img = F.hflip(sketch_img)
                positive_img = F.hflip(positive_img)
                negative_img = F.hflip(negative_img)

            sketch_img = self.train_transform(sketch_img)
            positive_img = self.train_transform(positive_img)
            negative_img = self.train_transform(negative_img)

            sample = {'sketch_img': sketch_img, 'sketch_path': sketch_path,
                      'positive_img': positive_img, 'positive_path': positive_sample,
                      'negative_img': negative_img, 'negative_path': negative_sample
                      }


        elif self.mode == 'Test':

            sketch_path = self.Test_Sketch[item]
            positive_sample = sketch_path.split('-')[0]

            sketch_img = self.test_transform(Image.open(self.sketch_root + sketch_path + '.png').convert('RGB'))
            positive_img = self.test_transform(Image.open(self.photo_root + positive_sample + '.jpg').convert('RGB'))

            sample = {'sketch_img': sketch_img, 'sketch_path': sketch_path,
                      'positive_img': positive_img, 'positive_path': positive_sample}

        return sample

    def __len__(self):
        if self.mode == 'Train':
            return len(self.Train_Sketch)
        elif self.mode == 'Test':
            return len(self.Test_Sketch)

def get_dataloader(hp):

    dataset_Train  = FGSBIR_Dataset(hp, mode = 'Train')
    dataloader_Train = data.DataLoader(dataset_Train, batch_size=hp.batchsize, shuffle=True,
                                         num_workers=int(hp.nThreads))

    dataset_Test  = FGSBIR_Dataset(hp, mode = 'Test')
    dataloader_Test = data.DataLoader(dataset_Test, batch_size=1, shuffle=False,
                                         num_workers=int(hp.nThreads))

    return dataloader_Train, dataloader_Test

def get_ransform(type):
    transform_list = []
    if type is 'Train':
        transform_list.extend([transforms.Resize(299)])
    elif type is 'Test':
        transform_list.extend([transforms.Resize(299)])
    transform_list.extend(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return transforms.Compose(transform_list)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Fine-Grained SBIR Model')

    parser.add_argument('--dataset_name', type=str, default='ShoeV2')
    parser.add_argument('--backbone_name', type=str, default='VGG', help='VGG / InceptionV3/ Resnet50')
    parser.add_argument('--pool_method', type=str, default='AdaptiveAvgPool2d',
                        help='AdaptiveMaxPool2d / AdaptiveAvgPool2d / AvgPool2d')
    parser.add_argument('--root_dir', type=str, default='/vol/research/sketchCV/datasets/')
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--nThreads', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--eval_freq_iter', type=int, default=100)
    parser.add_argument('--print_freq_iter', type=int, default=10)

    hp = parser.parse_args()
    print(hp)

    dataloader_Train, dataloader_Test = get_dataloader(hp)

    for batch in dataloader_Train:
        pics = ['sketch_img_1', 'sketch_img_2', 'sketch_img_3', 'positive_img', 'negative_img']
        for i in pics:
            print(batch[i].shape)
            tutils.save_image(batch[i], i+'.jpg', normalize=True)

        break

