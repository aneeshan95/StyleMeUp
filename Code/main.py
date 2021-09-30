import torch
import time
import random
import torch.nn.functional as F
from Networks import *
import learn2learn as l2l
from dataset import get_dataloader
import torch.nn as nn
from datetime import datetime
import argparse
from torch import optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, dataloader_Test):
    Image_Feature_ALL = []
    Image_Name = []
    Sketch_Feature_ALL = []
    Sketch_Name = []
    start_time = time.time()
    model.eval()
    for i_batch, sanpled_batch in enumerate(dataloader_Test):
        _, positive_feature, _ = model.encoder(sanpled_batch['positive_img'].to(device))
        _, sketch_feature, _ = model.encoder(sanpled_batch['sketch_img'].to(device))
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

    print('*' * 100, '\n\t Evaluation results: Time/sample(ms) :{:.2f} top1/top10(%): {:2.2f} / {:2.2f}\n'
          .format((time.time() - start_time) / len(Sketch_Name) * 1000, top1 * 100, top10 * 100) + '*' * 100)
    return top1, top10


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def train(model, positive, negative, sketch1, sketch2, mode='no_reg', regulariser=None):
    model.train()

    '''Encoding Inputs'''
    positive_dist, positive_inv, positive_var = model.encoder(positive)
    positive_z = positive_dist.rsample()

    negative_dist, negative_inv, negative_var = model.encoder(negative)
    negative_z = negative_dist.rsample()

    sketch1_dist, sketch1_inv, sketch1_var = model.encoder(sketch1)
    sketch1_z = sketch1_dist.rsample()

    sketch2_dist, sketch2_inv, sketch2_var = model.encoder(sketch2)
    sketch2_z = sketch2_dist.rsample()

    '''K-L Divergence Loss'''
    lambda_KL = 1.0

    positive_prior = torch.distributions.Normal(torch.zeros_like(positive_dist.mean), torch.ones_like(positive_dist.stddev))
    positive_KL = torch.distributions.kl_divergence(positive_dist, positive_prior).mean()

    negative_prior = torch.distributions.Normal(torch.zeros_like(negative_dist.mean), torch.ones_like(negative_dist.stddev))
    negative_KL = torch.distributions.kl_divergence(negative_dist, negative_prior).mean()

    sketch1_prior = torch.distributions.Normal(torch.zeros_like(sketch1_dist.mean), torch.ones_like(sketch1_dist.stddev))
    sketch1_KL = torch.distributions.kl_divergence(sketch1_dist, sketch1_prior).mean()

    sketch2_prior = torch.distributions.Normal(torch.zeros_like(sketch2_dist.mean), torch.ones_like(sketch2_dist.stddev))
    sketch2_KL = torch.distributions.kl_divergence(sketch2_dist, sketch2_prior).mean()

    total_KL = (positive_KL + negative_KL + sketch1_KL + sketch2_KL) / 4.0 * lambda_KL

    '''Self - Reconstruction Loss'''
    lambda_self_rec_loss = 1.0

    positive_recons = model.decoder(positive_z + positive_inv)
    positive_rec_loss = F.mse_loss(positive, positive_recons, reduction='mean') 

    negative_recons = model.decoder(negative_z + negative_inv)
    negative_rec_loss = F.mse_loss(negative, negative_recons, reduction='mean') 

    sketch1_recons = model.decoder(sketch1_z + sketch1_inv)
    sketch1_rec_loss = F.mse_loss(sketch1, sketch1_recons, reduction='mean') 

    sketch2_recons = model.decoder(sketch2_z + sketch2_inv)
    sketch2_rec_loss = F.mse_loss(sketch2, sketch2_recons, reduction='mean') 

    total_self_rec_loss = (positive_rec_loss + negative_rec_loss + sketch1_rec_loss + sketch2_rec_loss) / 4.0\
                          * lambda_self_rec_loss

    ''' Cross-style reconstruction loss'''
    lamba_cross_rec_loss = 1.0

    sketch12_recons = model.decoder(sketch1_inv + sketch2_z) # taking inv of 1 style and var of the other
    sketch12_rec_loss = F.mse_loss(sketch2, sketch12_recons, reduction='mean') 

    sketch21_recons = model.decoder(sketch2_inv + sketch1_z) # taking inv of 1 style and var of the other
    sketch21_rec_loss = F.mse_loss(sketch1, sketch21_recons, reduction='mean') 

    total_cross_style_rec_loss = (sketch21_rec_loss + sketch12_rec_loss) / 2.0 * lamba_cross_rec_loss

    ''' Cross-domain reconstruction loss'''
    lamba_cross_rec_loss = 1.0

    sketch1P_rec_loss = F.mse_loss(positive, sketch1_recons, reduction='mean') 
    sketch2P_rec_loss = F.mse_loss(positive, sketch2_recons, reduction='mean') 
    Psketch1_rec_loss = F.mse_loss(sketch2, positive_recons, reduction='mean') 
    Psketch2_rec_loss = F.mse_loss(sketch1, positive_recons, reduction='mean') 

    total_cross_domain_rec_loss = (sketch1P_rec_loss + sketch2P_rec_loss + Psketch1_rec_loss + Psketch2_rec_loss) \
                           / 4.0 * lamba_cross_rec_loss

    '''Triplet Loss'''
    lambda_triplet = 1.0
    positive_feature = torch.stack([positive_inv, positive_inv], dim=0)
    negative_feature = torch.stack([negative_inv, negative_inv], dim=0)
    sketch_feature = torch.stack([sketch1_inv, sketch2_inv], dim=0)
    triplet = tripletloss(sketch_feature, positive_feature, negative_feature) * lambda_triplet

    positive_feature = torch.stack([positive_z + positive_inv, positive_z + positive_inv], dim=0)
    negative_feature = torch.stack([negative_z + negative_inv, negative_z + negative_inv], dim=0)
    sketch_feature = torch.stack([sketch1_z + sketch1_inv, sketch2_z + sketch2_inv], dim=0)
    triplet += tripletloss(sketch_feature, positive_feature, negative_feature) * lambda_triplet

    total_loss = total_KL + total_cross_style_rec_loss + total_cross_domain_rec_loss + triplet
    reg_loss = 0
    if mode == 'reg':
        lambda_reg_loss = 1.0
        reg_loss = regulariser(torch.cat([model.encoder.invar.weight, model.encoder.invar.bias.unsqueeze(1)], dim=1))
        total_loss += reg_loss * lambda_reg_loss

    # nn.utils.clip_grad_norm(model.parameters(), 1.0) # self.hp.grad_clip =1.0

    return total_KL, total_self_rec_loss, total_cross_style_rec_loss, total_cross_domain_rec_loss, triplet, \
           reg_loss, total_loss

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Fine-Grained SBIR Model')

    parser.add_argument('--dataset_name', type=str, default='ShoeV2')
    parser.add_argument('--backbone_name', type=str, default='VGG', help='VGG / InceptionV3/ Resnet50')
    parser.add_argument('--pool_method', type=str, default='AdaptiveMaxPool2d',
                        help='AdaptiveMaxPool2d / AdaptiveAvgPool2d / AvgPool2d')
    parser.add_argument('--root_dir', type=str, default='../Datasets/')
    parser.add_argument('--batchsize', type=int, default=12)
    parser.add_argument('--nThreads', type=int, default=8)
    parser.add_argument('--meta_lr', type=float, default=0.0001)
    parser.add_argument('--fast_lr', type=float, default=0.005)
    parser.add_argument('--order', type=bool, default='2nd')
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--eval_freq_iter', type=int, default=20)
    parser.add_argument('--print_freq_iter', type=int, default=10)
    parser.add_argument('--remarks', type=str, default='VAE-meta with Reg.')

    hp = parser.parse_args()
    dataloader_Train, dataloader_Test = get_dataloader(hp)
    print(hp)
    print('Meta-FG-VAE approach with Meta-regulariser. Designed only for FIne-grained. Time of run: ', 
            datetime.now().strftime("%b-%d-%Y %H:%M:%S"))

    model = VAE_Network()
    model.to(device)
    Reg_model = Regulariser()
    Reg_model.to(device)

    tripletloss = nn.TripletMarginLoss(margin=0.2)    
    hp.order = '1st'
    print('No. of trainable parameters :', get_n_params(model))
    model = l2l.algorithms.MAML(model, hp.fast_lr, first_order=(hp.order == '1st'))
    optimizer_model = optim.Adam(model.parameters(), hp.meta_lr)
    optimizer_reg = optim.Adam(Reg_model.parameters(), hp.meta_lr)

    with torch.no_grad():
        top1_eval, top10_eval = 0,0 #evaluate(model, dataloader_Test)

    adaptation_steps = 1
    step_count, top1, top10 = 0, top1_eval, top10_eval

    # main training paradigm
    for i_epoch in range(hp.max_epoch):
        optimizer_model.zero_grad()
        optimizer_reg.zero_grad()
        for i_batch, batch in enumerate(dataloader_Train):
            step_count += 1
            meta_error = 0.0
            batch_time = time.time()

            for meta_batch in batch:

                # 1 step update
                learner = model.clone()

                '''Assignment'''
                all_style = [1, 2, 3]
                test_style = random.randint(1, 3)
                all_style.remove(test_style)

                positive = meta_batch['positive_img'].unsqueeze(0).to(device)
                negative = meta_batch['negative_img'].unsqueeze(0).to(device)
                sketch1 = meta_batch['sketch_img_' + str(all_style[0])].unsqueeze(0).to(device)
                sketch2 = meta_batch['sketch_img_' + str(all_style[1])].unsqueeze(0).to(device)
                sketch3 = meta_batch['sketch_img_' + str(test_style)].unsqueeze(0).to(device)

                '''Meta-training + regularising'''
                KL, self_recons, cross_style_recons, cross_domain_recons, triplet, reg_loss, total = \
                    train(learner, positive, negative, sketch1, sketch2, mode='reg', regulariser=Reg_model) 
                learner.adapt(total)

                print('Epoch: {} Step: {} KL: {:3.2f} Self-Recons: {:3.2f} C/S-Recons {:3.2f}'
                      ' C/D-Recons {:3.2f} Triplet: {:3.2f} Reg loss : {:3.2f} Total: {:3.2f} '
                      'top1/top10(%): {:2.2f} / {:2.2f}'
                      .format(i_epoch, step_count, KL, self_recons, cross_style_recons, cross_domain_recons,
                              triplet, reg_loss, total, top1 * 100, top10 * 100))
                # -------------------------------------------------------------------------------------------------

                '''Meta-test starts  -- no reg, just VAE loss'''
                KL, self_recons, cross_style_recons, cross_domain_recons, triplet, _, total = \
                    train(learner, positive, negative, sketch2, sketch3, mode='no_reg' )
                if step_count % hp.print_freq_iter == 0 :
                    print('Epoch: {} Step: {} KL: {:3.2f} Self-Recons: {:3.2f} C/S-Recons {:3.2f}'
                      ' C/D-Recons {:3.2f} Triplet: {:3.2f} Total: {:3.2f} '
                      .format(i_epoch, step_count, KL, self_recons, cross_style_recons, cross_domain_recons,
                              triplet, total))
                meta_loss = total
                meta_error += meta_loss.item()
                meta_loss.backward()
                # nn.utils.clip_grad_norm_(learner.parameters(), 0.1)

            if step_count % hp.print_freq_iter == 0 :
                print('\nTrain_Epoch: {:3.0f} Batch_num: {:3.0f} Time(s): {:3.3f} \tMeta_loss: {:3.2f}'
                  ' top1/top10(%): {:2.2f} / {:2.2f} \n'
                  .format(i_epoch, i_batch, time.time() - batch_time, meta_error / hp.batchsize, top1 * 100,
                          top10 * 100))
            for p in model.parameters():
                p.grad.data.mul_(1.0 / hp.batchsize)
            optimizer_model.step()        # Updating Model
            optimizer_reg.step()          # Updating Regulariser

            if step_count % hp.eval_freq_iter == 0:
                # collecting retrieval accurray
                with torch.no_grad():
                    top1_eval, top10_eval = evaluate(model, dataloader_Test)

                if top1_eval > top1:
                    torch.save(model.state_dict(), hp.backbone_name + '_' + hp.dataset_name + '_model_best.pth')
                    top1, top10 = top1_eval, top10_eval
                    print('Model Updated')
            pass

        pass
