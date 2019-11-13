import torch.nn as nn
from inpaint_modules import *
from collections import OrderedDict
from utils import *


class CoarseNet(nn.Module):
    def __init__(self,in_ch=5,cnum=48):
        super(CoarseNet,self).__init__()
        self.conv1 = Gated_Conv(in_ch=in_ch,out_ch=cnum,ksize=5)
        self.conv2_down = Gated_Conv(in_ch=cnum,out_ch=2*cnum,stride=2)
        self.conv3 = Gated_Conv(in_ch=2*cnum,out_ch=2*cnum)
        self.conv4_down = Gated_Conv(in_ch=2*cnum,out_ch=4*cnum,stride=2)
        self.conv5 = Gated_Conv(in_ch=4*cnum,out_ch=4*cnum)
        self.conv6 = Gated_Conv(in_ch=4*cnum,out_ch=4*cnum)

        self.conv7 = Gated_Conv(in_ch=4*cnum,out_ch=4*cnum,rate=2)
        self.conv8 = Gated_Conv(in_ch=4 * cnum, out_ch=4 * cnum, rate=4)
        self.conv9 = Gated_Conv(in_ch=4 * cnum, out_ch=4 * cnum, rate=8)
        self.conv10 = Gated_Conv(in_ch=4 * cnum, out_ch=4 * cnum, rate=16)

        self.conv11 = Gated_Conv(in_ch=4 * cnum, out_ch=4 * cnum)
        self.conv12 = Gated_Conv(in_ch=4 * cnum, out_ch=4 * cnum)

        self.conv13_up = Gated_Deconv(in_ch=4*cnum,out_ch=2*cnum)
        self.conv14 = Gated_Conv(in_ch=2*cnum,out_ch=2*cnum)
        self.conv15_up = Gated_Deconv(in_ch=2*cnum,out_ch=cnum)
        self.conv16 = Gated_Conv(in_ch=cnum,out_ch=cnum//2)

        self.conv17 = nn.Conv2d(in_channels=cnum//2,out_channels=3,kernel_size=3,stride=1,padding=1)


    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.conv2_down(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4_down(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        x9 = self.conv9(x8)
        x10 = self.conv10(x9)
        x11 = self.conv11(x10)
        x12 = self.conv12(x11)
        x13 = self.conv13_up(x12)
        x14 = self.conv14(x13)
        x15 = self.conv15_up(x14)
        x16 = self.conv16(x15)
        x17 = self.conv17(x16)
        x_stage1 = F.tanh(x17)
        return x_stage1

class RefineNet(nn.Module):
    def __init__(self,in_ch=3,cnum=48):
        super(RefineNet,self).__init__()
        #conv branch
        xconv_layer = OrderedDict()
        xconv_layer['xconv1'] = Gated_Conv(in_ch=in_ch,out_ch=cnum,ksize=5)
        xconv_layer['xconv2_down'] = Gated_Conv(in_ch=cnum,out_ch=cnum,stride=2)
        xconv_layer['xconv3'] =  Gated_Conv(in_ch=cnum,out_ch=2*cnum)
        xconv_layer['xconv4_down'] = Gated_Conv(in_ch=2*cnum,out_ch=2*cnum,stride=2)
        xconv_layer['xconv5'] = Gated_Conv(in_ch=2*cnum,out_ch=4*cnum)
        xconv_layer['xconv6'] = Gated_Conv(in_ch=4*cnum,out_ch=4*cnum)

        xconv_layer['xconv7_atrous']  = Gated_Conv(in_ch=4*cnum,out_ch=4*cnum,rate=2)
        xconv_layer['xconv8_atrous'] = Gated_Conv(in_ch=4 * cnum, out_ch=4 * cnum, rate=4)
        xconv_layer['xconv9_atrous'] = Gated_Conv(in_ch=4 * cnum, out_ch=4 * cnum, rate=8)
        xconv_layer['xconv10_atrous'] = Gated_Conv(in_ch=4 * cnum, out_ch=4 * cnum, rate=16)

        self.xlayer = nn.Sequential(xconv_layer)

        #attention brach
        pmconv_layer1 = OrderedDict()
        pmconv_layer1['pmconv1'] = Gated_Conv(in_ch=in_ch,out_ch=cnum,ksize=5)
        pmconv_layer1['pmconv2_down'] = Gated_Conv(in_ch=cnum,out_ch=cnum,stride=2)
        pmconv_layer1['pmconv3'] = Gated_Conv(in_ch=cnum,out_ch=2*cnum)
        pmconv_layer1['pmconv4_down'] = Gated_Conv(in_ch=2*cnum, out_ch=4*cnum, stride=2)
        pmconv_layer1['pmconv5'] = Gated_Conv(in_ch=4*cnum,out_ch=4*cnum)
        pmconv_layer1['pmconv6'] = Gated_Conv(in_ch=4 * cnum, out_ch=4 * cnum,activation=nn.ReLU())
        self.pmlayer1 = nn.Sequential(pmconv_layer1)

        self.CA = Contextual_Attention(rate=2)

        pmconv_layer2 = OrderedDict()
        pmconv_layer2['pmconv9'] = Gated_Conv(in_ch=4*cnum,out_ch=4*cnum)
        pmconv_layer2['pmconv10'] = Gated_Conv(in_ch=4*cnum,out_ch=4*cnum)
        self.pmlayer2 = nn.Sequential(pmconv_layer2)

        #confluent branch
        allconv_layer = OrderedDict()
        allconv_layer['allconv11'] = Gated_Conv(in_ch=8*cnum,out_ch=4*cnum)
        allconv_layer['allconv12'] = Gated_Conv(in_ch=4 * cnum, out_ch=4 * cnum)
        allconv_layer['allconv13_up'] = Gated_Deconv(in_ch=4 * cnum, out_ch=2 * cnum)
        allconv_layer['allconv14'] = Gated_Conv(in_ch=2 * cnum, out_ch=2 * cnum)
        allconv_layer['allconv15_up'] = Gated_Deconv(in_ch=2 * cnum, out_ch=cnum)
        allconv_layer['allconv16'] = Gated_Conv(in_ch=cnum, out_ch=cnum//2)
        allconv_layer['allconv17'] = nn.Conv2d(in_channels=cnum//2,out_channels=3,kernel_size=3,padding=1)
        allconv_layer['tanh'] = nn.Tanh()
        self.colayer = nn.Sequential(allconv_layer)

    def forward(self, xin, mask):

        x1 = self.xlayer(xin)
        x_hallu = x1

        x2 = self.pmlayer1(xin)
        mask_s = self.resize_mask_like(mask,x2)
        x3,offset_flow = self.CA(x2,x2,mask_s)
        x4 = self.pmlayer2(x3)
        pm = x4

        x5 = torch.cat((x_hallu,pm),dim=1)
        x6 = self.colayer(x5)
        x_stage2 = x6

        return x_stage2,offset_flow

    def resize_mask_like(self,mask,x):
        sizeh = x.shape[2]
        sizew = x.shape[3]
        return down_sample(mask,size=(sizeh,sizew))

class CAGenerator(nn.Module):
    def __init__(self,in_ch=5,cnum=48,):
        super(CAGenerator,self).__init__()
        self.stage_1 = CoarseNet(in_ch=in_ch,cnum=cnum)
        self.stage_2 = RefineNet(in_ch=3,cnum=cnum)

    def forward(self,xin,mask):
        stage1_out = self.stage_1(xin)
        stage2_in = stage1_out * mask + xin[:,0:3,:,:] * (1. - mask)
        stage2_out,offset_flow = self.stage_2(stage2_in,mask)

        return stage1_out,stage2_out,offset_flow


class SNDiscriminator(nn.Module):
    def __init__(self,in_ch=5,cnum=64):
        super(SNDiscriminator,self).__init__()

        disconv_layer = OrderedDict()
        disconv_layer['conv1'] = SN_Conv(in_ch=in_ch,out_ch=cnum,ksize=5,stride=2)
        disconv_layer['conv2'] = SN_Conv(in_ch=cnum, out_ch=2*cnum, ksize=5, stride=2)
        disconv_layer['conv3'] = SN_Conv(in_ch=2*cnum, out_ch=4*cnum, ksize=5, stride=2)
        disconv_layer['conv4'] = SN_Conv(in_ch=4 * cnum, out_ch=4 * cnum, ksize=5, stride=2)
        disconv_layer['conv5'] = SN_Conv(in_ch=4 * cnum, out_ch=4 * cnum, ksize=5, stride=2)
        disconv_layer['conv6'] = SN_Conv(in_ch=4 * cnum, out_ch=4 * cnum, ksize=5, stride=2)
        self.dislayer = nn.Sequential(disconv_layer)

    def forward(self,x):
        x1 = self.dislayer(x)
        out = x1.view(x1.shape[0],-1)
        return out


if __name__ == '__main__':
    G = CAGenerator()
    D = SNDiscriminator()
    for key,_ in D.named_parameters():
        print(key)
