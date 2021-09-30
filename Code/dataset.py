import argparse
import torch
import pickle
import torch.utils.data as data
import torchvision.transforms as transforms
import os
from random import randint
from PIL import Image
import random
import torchvision.utils as tutils
import torchvision.transforms.functional as F
from rasterize import rasterize_Sketch



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FGSBIR_Dataset(data.Dataset):
    def __init__(self, hp, mode):

        self.hp = hp
        self.mode = mode
        coordinate_path = os.path.join(hp.root_dir, hp.dataset_name , hp.dataset_name + '_Coordinate')
        # /vol/research/sketchCV/datasets/ShoeV2/ShoeV2_Coordinate
        self.root_dir = os.path.join(hp.root_dir, hp.dataset_name)
        with open(coordinate_path, 'rb') as fp:
            self.Coordinate = pickle.load(fp)

        # self.Train_Sketch = [x for x in self.Coordinate if 'train' in x]
        # self.Test_Sketch = [x for x in self.Coordinate if 'test' in x]

        self.Train_Sketch = [[],[],[]]
        self.Test_Sketch = []

        # example: self.Train_Sketch[0] = /train/00140128_v1_1
        for x in self.Coordinate:
            if x.split('_')[-1] not in ['1', '2', '3']:                          # limiting to 3 instances per photo
                continue
            elif not(x[:-1]+'1' in self.Coordinate and x[:-1]+'2' in self.Coordinate and x[:-1]+'3' in self.Coordinate):
                continue                                               # rejecting pics without sufficient instances


            elif 'train' in x:
                self.Train_Sketch[int(x[-1])-1].append(x)
            elif 'test' in x:
                self.Test_Sketch.append(x)

        self.transform = get_ransform()



    def __getitem__(self, item):
        sample  = {}
        if self.mode == 'Train':
            sketch_1 = self.Train_Sketch[0][item]
            sketch_2 = self.Train_Sketch[1][item]
            sketch_3 = self.Train_Sketch[2][item]

            positive_sample = sketch_1[sketch_1.rfind('/')+1: sketch_1.rfind('_')]
            positive_path = os.path.join(self.root_dir, 'photo', positive_sample + '.png')
            positive_img = Image.open(positive_path).convert('RGB')

            files = os.listdir(self.root_dir + '/photo/')                  # taking a negative sample
            files.remove(positive_sample + '.png')
            negative_sample = files[random.sample(range(len(files)), 1)[0]][:-4]
            negative_path = os.path.join(self.root_dir, 'photo', negative_sample + '.png')
            negative_img = Image.open(negative_path).convert('RGB')

            vector_x = self.Coordinate[sketch_1]                                            #S1
            sketch_img = rasterize_Sketch(vector_x)
            sketch_img_1 = Image.fromarray(sketch_img).convert('RGB')

            vector_x = self.Coordinate[sketch_2]                                            #S2
            sketch_img = rasterize_Sketch(vector_x)
            sketch_img_2 = Image.fromarray(sketch_img).convert('RGB')

            vector_x = self.Coordinate[sketch_3]                                            #S3
            sketch_img = rasterize_Sketch(vector_x)
            sketch_img_3 = Image.fromarray(sketch_img).convert('RGB')

            n_flip = random.random()
            if n_flip > 0.5:
                sketch_img_1 = F.hflip(sketch_img_1)
                sketch_img_2 = F.hflip(sketch_img_2)
                sketch_img_3 = F.hflip(sketch_img_3)
                positive_img = F.hflip(positive_img)
                negative_img = F.hflip(negative_img)

            sketch_img_1 = self.transform(sketch_img_1)
            sketch_img_2 = self.transform(sketch_img_2)
            sketch_img_3 = self.transform(sketch_img_3)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

            sample = {'sketch_img_1': sketch_img_1, 'sketch_img_2': sketch_img_2, 'sketch_img_3': sketch_img_3,
                      'sketch_path': sketch_1,
                      'positive_img': positive_img, 'positive_path': positive_sample,
                      'negative_img': negative_img, 'negative_path': negative_sample
                      }



        elif self.mode == 'Test':

            sketch_path = self.Test_Sketch[item]
            vector_x = self.Coordinate[sketch_path]
            sketch_img = rasterize_Sketch(vector_x)
            sketch_img = self.transform(Image.fromarray(sketch_img).convert('RGB'))

            positive_sample = '_'.join(self.Test_Sketch[item].split('/')[-1].split('_')[:-1])
            positive_path = os.path.join(self.root_dir, 'photo', positive_sample + '.png')
            positive_img = self.transform(Image.open(positive_path).convert('RGB'))

            sample = {'sketch_img': sketch_img, 'sketch_path': sketch_path,
                      'positive_img': positive_img, 'positive_path': positive_sample}

        return sample


    def __len__(self):
        if self.mode == 'Train':
            return len(self.Train_Sketch[0])
        elif self.mode == 'Test':
            return len(self.Test_Sketch)

def collate_self(batch):
    return batch

def get_negative(positive_sample, filepath):

    '''
    :param positive_sample: format : 1031000079   without extension of png
    :param filepath: phot directory
    :return: negative sample : format 1031000078   without extension of png
    '''

    files = os.listdir(filepath)
    files.remove(positive_sample+'.png')
    neg_index = random.sample(range(len(files)), 1)

    return files[neg_index][:-4]


def get_dataloader(hp):

    dataset_Train  = FGSBIR_Dataset(hp, mode = 'Train')
    dataloader_Train = data.DataLoader(dataset_Train, batch_size=hp.batchsize, shuffle=True,
                                         num_workers=int(hp.nThreads), collate_fn=collate_self)

    dataset_Test  = FGSBIR_Dataset(hp, mode = 'Test')
    dataloader_Test = data.DataLoader(dataset_Test, batch_size=1, shuffle=False,
                                         num_workers=int(hp.nThreads))

    return dataloader_Train, dataloader_Test

def get_ransform():
    # transform_list = [transforms.Resize(299), transforms.ToTensor(),
    #                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    transform_list = [transforms.Resize((256, 256)), transforms.ToTensor(),
                      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]

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

        k=2 # just to create a breakpoint

    print('done')

