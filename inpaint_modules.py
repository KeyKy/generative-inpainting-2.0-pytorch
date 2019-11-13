import torch.nn as nn
import torch
import torch.nn.functional as F
from utils import  *
from torch.nn import Parameter


class Gated_Conv(nn.Module):
    def __init__(self,in_ch,out_ch,ksize=3,stride=1,rate=1,activation=nn.ELU()):
        super(Gated_Conv,self).__init__()
        padding = int(rate*(ksize-1)/2)
        self.conv = nn.Conv2d(in_ch,2*out_ch,kernel_size=ksize,stride=stride,padding=padding,dilation=rate)
        self.activation = activation


    def forward(self, x):
        raw = self.conv(x)
        x1 = raw.split(int(raw.shape[1]/2),dim=1)
        gate = F.sigmoid(x1[0])
        out = self.activation(x1[1])*gate
        return out

class Gated_Deconv(nn.Module):
    def __init__(self,in_ch,out_ch,ksize=3,stride=1,rate=1,activation=nn.ELU()):
        super(Gated_Deconv,self).__init__()
        self.up_sample = nn.Upsample(scale_factor=2,mode='nearest')
        self.conv = Gated_Conv(in_ch=in_ch,out_ch=out_ch,ksize=ksize,stride=stride,rate=rate,activation=activation)

    def forward(self, x):
        x = self.up_sample(x)
        out = self.conv(x)
        return out

class SN_Conv(nn.Module):
    def __init__(self,in_ch,out_ch,ksize=3,stride=1,rate=1,activation=nn.LeakyReLU()):
        super(SN_Conv,self).__init__()
        padding = int(rate * (ksize - 1) / 2)
        conv = nn.Conv2d(in_ch,out_ch, kernel_size=ksize, stride=stride, padding=padding, dilation=rate)
        self.snconv = SpectralNorm(conv)
        self.activation = activation

    def forward(self,x):
        x1 = self.snconv(x)
        if self.activation is not None:
            x1 = self.activation(x1)

        return x1



class Contextual_Attention(nn.Module):
    def __init__(self,ksize=3,stride=1,rate=2,fuse_k=3,padding=1,softmax_scale=10.,training=True,fuse=True):
        super(Contextual_Attention,self).__init__()
        self.padding = nn.ZeroPad2d(padding)
        self.ksize = ksize
        self.stride = stride
        self.rate = rate
        self.fuse_k = fuse_k
        self.softmax_scale = softmax_scale
        self.training = training
        self.fuse = fuse
        self.up_sample = nn.Upsample(scale_factor=self.rate,mode='nearest')

    def extract_patches(self,x,ksize=3,stride=1):
        x = self.padding(x)
        out = x.unfold(2,ksize,stride).unfold(3,ksize,stride)
        return out

    def forward(self, f, b, mask=None):
        """
        Contextual attention layer implementation.
        Contextual attention is first introduced in publication:
        Generative Image Inpainting with Contextual Attention, Yu et al.
        Modified from https://github.com/WonwoongCho/Generative-Inpainting-pytorch/blob/daef8659cb0e15359f32a63c2159f17d75a555e6/model_module.py
        which has a few mistakes.
        :param f:foreground features B x 128 x 64 x 64
        :param b: background features B x 128 64 x 64
        :param mask:
        :return:
        """
        #print('f_shape:',f.shape,'b_shape:',b.shape)
        #get shapes
        raw_fs = f.size()  # B x 128 x 64 x 64
        raw_int_fs = list(f.size())
        raw_int_bs = list(b.size())

        #extract patches from background with stride and rate
        kernel = 2*self.rate

        raw_w = self.extract_patches(b,ksize=kernel,stride=self.rate)  # B x 128 x 32 x 32 x 4 x 4
        raw_w = raw_w.contiguous().view(raw_int_bs[0],raw_int_bs[1],-1,kernel,kernel)  # B x 128 x 1024 x 4 x 4
        raw_w = raw_w.permute(0,2,1,3,4)    # B x 1024 x 128 x 4 x 4

        # downscaling foreground option: downscaling both foreground and
        # background for matching and use original background for reconstruction.
        f = down_sample(f,scale_factor=1/self.rate,mode='nearest')
        b = down_sample(b,scale_factor=1/self.rate,mode='nearest')

        fs = f.size()  # B x 128 x 32 x 32
        int_fs = list(f.size())
        f_groups = torch.split(f,1,dim=0)

        bs = b.size()
        int_bs = list(b.size())
        w = self.extract_patches(b)  # B x 128 x 32 x 32 x 3 x 3
        w = w.contiguous().view(int_bs[0],int_bs[1],-1,self.ksize,self.ksize) # B x 128 x 1024 x 4 x 4
        w = w.permute(0,2,1,3,4)

        #process mask
        if mask is not None:
            mask = down_sample(mask,scale_factor=1/self.rate,mode='nearest')
        else:
            mask = torch.zeros([1,1,bs[2],bs[3]])

        m = self.extract_patches(mask)
        #print('M_shape:',m.shape)
        m = m.contiguous().view(1,1,-1,self.ksize,self.ksize)
        #print('m_shape:',m.shape)
        m = m[0]    # 1 x 1024 x 3 x 3
        m = reduce_mean(m)
        mm = m.eq(0.).float() # 1 x 1024 x 1 x 1
        #print('mm_shape',mm.shape)

        w_groups = torch.split(w,1,dim=0)
        raw_w_groups = torch.split(raw_w,1,dim=0)
        y = []
        offsets = []
        k = self.fuse_k
        scale = self.softmax_scale
        fuse_weight = Variable(torch.eye(k).view(1,1,k,k)).cuda()

        for xi,wi,raw_wi in zip(f_groups,w_groups,raw_w_groups):
            '''
            O => output channel as a conv filter
            I => input channel as a conv filter
            xi : separated tensor along batch dimension of front; (B=1, C=128, H=32, W=32)
            wi : separated patch tensor along batch dimension of back; (B=1, O=32*32, I=128, KH=3, KW=3)
            raw_wi : separated tensor along batch dimension of back; (B=1, I=32*32, O=128, KH=4, KW=4)
            '''
            #conv for compare
            wi = wi[0]
            escape_NaN = Variable(torch.FloatTensor([1e-4])).cuda()
            wi_normed = wi / torch.max(l2_norm(wi),escape_NaN)
            yi = F.conv2d(xi,wi_normed,stride=1,padding=1)  # yi => (B=1, C=32*32, H=32, W=32)
            #print('yi_shape:',yi.shape)

            #conv implementation for fuse scores to encourage large patches
            if self.fuse:
                yi = yi.view(1, 1, fs[2] * fs[3],
                         bs[2] * bs[3])  # make all of depth to spatial resolution, (B=1, I=1, H=32*32, W=32*32)
                yi = F.conv2d(yi, fuse_weight, stride=1, padding=1)  # (B=1, C=1, H=32*32, W=32*32)

                yi = yi.contiguous().view(1, fs[2], fs[3], bs[2], bs[3])  # (B=1, 32, 32, 32, 32)
                yi = yi.permute(0, 2, 1, 4, 3)
                yi = yi.contiguous().view(1, 1, fs[2] * fs[3], bs[2] * bs[3])

                yi = F.conv2d(yi, fuse_weight, stride=1, padding=1)
                yi = yi.contiguous().view(1, fs[3], fs[2], bs[3], bs[2])
                yi = yi.permute(0, 2, 1, 4, 3)

            yi = yi.contiguous().view(1, bs[2] * bs[3], fs[2], fs[3])  # (B=1, C=32*32, H=32, W=32)
            #print('bs:',bs,'fs:',fs)
            #print('yi_shape:',yi.shape,'mm_shape:',mm.shape)

            # softmax to match
            yi = yi * mm  # mm => (1, 32*32, 1, 1)
            yi = F.softmax(yi * scale, dim=1)
            yi = yi * mm  # mask

            _, offset = torch.max(yi, dim=1)  # argmax; index
            division = torch.div(offset, fs[3]).long()
            offset = torch.stack([division, torch.div(offset, fs[3]) - division], dim=-1)

            # deconv for patch pasting
            # 3.1 paste center
            wi_center = raw_wi[0]
            yi = F.conv_transpose2d(yi, wi_center, stride=self.rate, padding=1) / 4.  # (B=1, C=128, H=64, W=64)
            y.append(yi)
            offsets.append(offset)

        y = torch.cat(y, dim=0)  # back to the mini-batch
        y.contiguous().view(raw_int_fs)
        offsets = torch.cat(offsets, dim=0)
        offsets = offsets.view([int_bs[0]] + [2] + int_bs[2:])

        # case1: visualize optical flow: minus current position
        h_add = Variable(torch.arange(0, float(bs[2]))).cuda().view([1, 1, bs[2], 1])
        h_add = h_add.expand(bs[0], 1, bs[2], bs[3])
        w_add = Variable(torch.arange(0, float(bs[3]))).cuda().view([1, 1, 1, bs[3]])
        w_add = w_add.expand(bs[0], 1, bs[2], bs[3])

        offsets = offsets - torch.cat([h_add, w_add], dim=1).long()

        # to flow image
        flow = torch.from_numpy(flow_to_image(offsets.permute(0, 2, 3, 1).cpu().data.numpy()))
        flow = flow.permute(0, 3, 1, 2)

        if self.rate != 1:
            flow = self.up_sample(flow)

        return y,flow


class SpectralNorm(nn.Module):
    '''
    spectral normalization,modified from https://github.com/christiancosgrove/pytorch-spectral-normalization-gan/blob/master/spectral_normalization.py

    '''
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)



