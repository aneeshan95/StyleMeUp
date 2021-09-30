import torch
import time
import numpy as np
from sklearn.metrics import average_precision_score
from scipy.spatial.distance import cdist
import torch.nn.functional as F

import random
from Networks import VGG_Network
import learn2learn as l2l
from dataset import get_dataloader
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import argparse
from torch import optim


def SBIR(model, datloader_Test):
    Image_Feature_ALL = []
    Image_Name = []
    Sketch_Feature_ALL = []
    Sketch_Name = []
    acc_cls_sk = []
    acc_cls_im = []
    start_time = time.time()

    model.eval()
    for i_batch, sanpled_batch in enumerate(datloader_Test):
        sketch_feature = model(sanpled_batch['sketch_img'].to(device))
        positive_feature = model(sanpled_batch['positive_img'].to(device))

        Sketch_Feature_ALL.extend(sketch_feature)
        Sketch_Name.extend(sanpled_batch['sketch_path'])

        for i_num, positive_name in enumerate(sanpled_batch['positive_path']):
            acc_cls_sk.append(sanpled_batch['sketch_path'][i_num].split('/')[0])
            if positive_name not in Image_Name:
                Image_Name.append(sanpled_batch['positive_path'][i_num])
                Image_Feature_ALL.append(positive_feature[i_num])
                acc_cls_im.append(sanpled_batch['positive_path'][i_num].split('/')[0])

    Image_Feature_ALL = torch.stack(Image_Feature_ALL)
    Sketch_Feature_ALL_np = torch.stack(Sketch_Feature_ALL).cpu().numpy()
    distance2 = cdist(Sketch_Feature_ALL_np, Image_Feature_ALL.cpu().numpy())
    sim = 1 / (1 + distance2)
    str_sim = (np.expand_dims(acc_cls_sk, axis=1) == np.expand_dims(acc_cls_im, axis=0)) * 1
    q_num = str_sim.shape[0]
    aps = [average_precision_score(str_sim[iq], sim[iq]) for iq in range(q_num)]
    mAP = np.mean(aps)

    print('Time to EValuate:{}'.format(time.time() - start_time))
    return mAP


def FGSBIR(model, dataloader_Test):
    Image_Feature_ALL = []
    Image_Name = []
    Sketch_Feature_ALL = []
    Sketch_Name = []
    start_time = time.time()
    model.eval()
    for i_batch, sanpled_batch in enumerate(dataloader_Test):
        sketch_feature  = model(sanpled_batch['sketch_img'].to(device))
        positive_feature= model(sanpled_batch['positive_img'].to(device))
        Sketch_Feature_ALL.extend(sketch_feature)
        Sketch_Name.extend(sanpled_batch['sketch_path'])

        for i_num, positive_name in enumerate(sanpled_batch['positive_path']):
            if positive_name not in Image_Name:
                Image_Name.append(sanpled_batch['positive_path'][i_num])
                Image_Feature_ALL.append(positive_feature[i_num])

    rank = torch.zeros(len(Sketch_Name))
    Image_Feature_ALL = torch.stack(Image_Feature_ALL)

    for num, sketch_feature in enumerate(Sketch_Feature_ALL):
        s_name = Sketch_Name[num]
        sketch_query_name = '_'.join(s_name.split('/')[-1].split('_')[:-1])
        position_query = Image_Name.index(sketch_query_name)

        distance = F.pairwise_distance(sketch_feature.unsqueeze(0), Image_Feature_ALL)
        target_distance = F.pairwise_distance(sketch_feature.unsqueeze(0),
                                              Image_Feature_ALL[position_query].unsqueeze(0))

        rank[num] = distance.le(target_distance).sum()

    top1 = rank.le(1).sum().numpy() / rank.shape[0]
    top10 = rank.le(10).sum().numpy() / rank.shape[0]

    print('*'*100,'\n\t Evaluation results: Time/sample(ms) :{:.2f} top1_score: {:.2f}% top10_score: {:.2f}%\n'
          .format((time.time() - start_time)/len(Sketch_Name)*1000, top1*100, top10*100)+'*'*100)
    return top1, top10