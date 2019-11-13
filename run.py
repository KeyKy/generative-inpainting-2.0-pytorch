import torch
from dataloader import get_loader
from inpaint_model import *
import time
from utils import *
import torch.nn.functional as F
import numpy as np
import os
from config import config


class Run(object):
    def __init__(self,args):
        self.args = args

        self.data_loader = get_loader(batch_size=self.args.batch_size,FLAGS=self.args,dataset='CelebA',mode=self.args.mode)

        self.init_network()


    def init_network(self):
        #Models
        if self.args.pretrained_model_G:
            self.G = torch.load(os.path.join(self.args.model_save_path,self.args.pretrained_model_G))
            self.D = torch.load(os.path.join(self.args.model_save_path,self.args.pretrained_model_D))

        else:
            self.G = CAGenerator()
            self.D = SNDiscriminator()

        if self.args.cuda:
            self.G = self.G.cuda()
            self.D = self.D.cuda()

        #optimizer
        self.g_optimizer = torch.optim.Adam(self.G.parameters(),self.args.g_lr)
        self.d_optimizer = torch.optim.Adam(self.D.parameters(),self.args.d_lr)

        #loss


    def train(self):
        iters_per_epoch = len(self.data_loader)

        g_lr = self.args.g_lr
        d_lr = self.args.d_lr

        if self.args.pretrained_model_G:
            start = int(self.args.pretrained_model_G.split('_')[0]) + 1
        else:
            start = 1

        start_time = time.time()
        self.G.train()
        self.D.train()
        for epoch in range(start,self.args.num_epochs):
            for batch_index,real_images in enumerate(self.data_loader): #real_image: B x 3 x H x W

                if self.args.guided:
                    real_images,edge = real_images    #edge:a tensor with range (0~1) and shape(batch_size,1,W,H)
                    edge = (edge > self.args.edge_threshold).type(torch.float32)
                else:
                    edge = torch.zeros_like(real_images)[:,0:1,:,:].type(torch.float32)

                batch_size, height, width = real_images.shape[0], real_images.shape[2], real_images.shape[3]
                real_images = 2.*real_images - 1. #[-1,1]

                #generate mask ,1 represents masked point
                bbox = random_bbox(self.args)
                regular_mask = bbox2mask(self.args,bbox)
                irregular_mask = brush_stroke_mask(self.args)

                binary_mask = np.logical_or(regular_mask.astype(np.bool),irregular_mask.astype(np.bool)).astype(np.float32)
                binary_mask = torch.FloatTensor(binary_mask)

                batch_mask = binary_mask.repeat(batch_size,1,1,1)

                masked_edge = batch_mask * edge

                inverse_mask = 1. - batch_mask
                masked_images = real_images.clone() * inverse_mask
                data_input = torch.cat((masked_images,batch_mask,masked_edge),dim=1)

                if self.args.cuda:
                    data_input = data_input.cuda()
                    batch_mask = batch_mask.cuda()
                    masked_edge = masked_edge.cuda()
                    binary_mask = binary_mask.cuda()
                    inverse_mask = inverse_mask.cuda()
                    masked_images = masked_images.cuda()
                    real_images = real_images.cuda()

                stage_1,stage_2,offset_flow = self.G(data_input,binary_mask)


                batch_complete = stage_2 * batch_mask + masked_images * inverse_mask

                ae_loss = self.args.l1_loss_alpha * reduce_mean(torch.abs(real_images-stage_1),dim=[0,1,2,3]).view(-1)
                ae_loss +=self.args.l1_loss_alpha * reduce_mean(torch.abs(real_images-stage_2),dim=[0,1,2,3]).view(-1)

                batch_pos_neg = torch.cat((real_images,batch_complete),dim=0)
                if self.args.gan_with_mask:
                    batch_pos_neg = torch.cat((batch_pos_neg,batch_mask.repeat(2,1,1,1)),dim=1)

                if self.args.guided:
                    # conditional gan
                    batch_pos_neg = torch.cat((batch_pos_neg,masked_edge.repeat(2,1,1,1)),dim=1)
                else:
                    batch_pos_neg = torch.cat((batch_pos_neg, torch.zeros_like(masked_edge).repeat(2, 1, 1, 1)), dim=1)


                if self.args.gan == 'sngan':
                    pos_neg = self.D(batch_pos_neg)
                    pos,neg = torch.split(pos_neg,batch_size,dim=0)
                    self.g_loss,self.d_loss = self.gan_hinge_loss(pos,neg)

                else:
                    raise NotImplementedError('{} is not implemented.'.format(self.args.gan))

                self.g_loss += ae_loss

                self.backprop(G=True,D=True)

                print('epoch[{}] iter[{} / {}]'.format(epoch,batch_index,iters_per_epoch))

            # save model checkpoint
            if not os.path.exists(self.args.model_save_path):
                os.mkdir(self.args.model_save_path)
            torch.save(self.G,os.path.join(self.args.model_save_path,'{}_G_L1_{}.pth'.format(epoch,self.args.l1_loss_alpha)))
            torch.save(self.D,os.path.join(self.args.model_save_path, '{}_D_L1_{}.pth'.format(epoch, self.args.l1_loss_alpha)))

            # save sample image

    def backprop(self,G=True,D=True):
        if D:
            self.d_optimizer.zero_grad()
            self.d_loss.backward(retain_graph=G)
            self.d_optimizer.step()
        if G:
            self.g_optimizer.zero_grad()
            self.g_loss.backward()
            self.g_optimizer.step()


    def gan_hinge_loss(self,pos,neg,name='gan_hinge_loss'):
        #print('pos_shape:',pos.shape)
        #print('neg_shape:',neg.shape)
        hinge_pos = reduce_mean(F.relu(1-pos),dim=[0,1]).view(-1)
        hinge_neg = reduce_mean(F.relu(1+neg),dim=[0,1]).view(-1)
        d_loss = 0.5 * hinge_pos + 0.5 * hinge_neg
        g_loss = -reduce_mean(neg,dim=[0,1])
        return g_loss,d_loss


if __name__ == '__main__':
    args = config()
    runer = Run(args)
    if args.mode == 'train':
        runer.train()
