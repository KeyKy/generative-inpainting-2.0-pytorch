import torch.nn.functional as F
import torch
from torch.autograd import Variable
import numpy as np
import math
from PIL import Image,ImageDraw

def down_sample(x,size=None,scale_factor=None,mode='nearest'):

    if size is None and scale_factor is None:
        raise Exception('argument size or scale_factor must be sprcified.')

    if size is None:
        size = (int(scale_factor*x.shape[2]),int(scale_factor*x.shape[3]))

    #create coordinates
    h = torch.arange(0,size[0]).float() / (size[0]-1.) * 2. - 1.
    w = torch.arange(0,size[1]).float() / (size[1]-1.) * 2. - 1.

    # creat grid
    grid = torch.zeros(size[0],size[1],2)

    grid[:,:,0] = w.unsqueeze(0).repeat(size[0],1)
    grid[:,:,1] = h.unsqueeze(0).repeat(size[1],1).transpose(0,1)
    #expand to match batch size
    grid = grid.unsqueeze(0).repeat(x.shape[0],1,1,1)
    if x.is_cuda:
        grid = Variable(grid).cuda()
    return F.grid_sample(x,grid,mode=mode)

def reduce_mean(x,dim=[2,3]):
    for k,d in enumerate(dim):
        x = torch.mean(x,dim=d,keepdim=True)
    return x

def reduce_sum(x,dim=[0,2,3]):
    for k,d in enumerate(dim):
        x = torch.sum(x,dim=d,keepdim=True)
    return x

def l2_norm(x):
    x = x**2
    y = torch.sqrt(reduce_sum(x))
    return y

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


def flow_to_image(flow):
    """Transfer flow map to image.
    Part of code forked from flownet.
    """
    out = []
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    maxrad = -1
    for i in range(flow.shape[0]):
        u = flow[i, :, :, 0]
        v = flow[i, :, :, 1]
        idxunknow = (abs(u) > 1e7) | (abs(v) > 1e7)
        u[idxunknow] = 0
        v[idxunknow] = 0
        maxu = max(maxu, np.max(u))
        minu = min(minu, np.min(u))
        maxv = max(maxv, np.max(v))
        minv = min(minv, np.min(v))
        rad = np.sqrt(u ** 2 + v ** 2)
        maxrad = max(maxrad, np.max(rad))
        u = u/(maxrad + np.finfo(float).eps)
        v = v/(maxrad + np.finfo(float).eps)
        img = compute_color(u, v)
        out.append(img)
    return np.float32(np.uint8(out))

def compute_color(u,v):
    h, w = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0
    # colorwheel = COLORWHEEL
    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)
    rad = np.sqrt(u**2+v**2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a+1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(int)
    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0
    for i in range(np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)
        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))
    return img

def make_color_wheel():
    RY, YG, GC, CB, BM, MR = (15, 6, 4, 11, 13, 6)
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros([ncols, 3])
    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255
    return colorwheel

COLORWHEEL = make_color_wheel()

def random_bbox(FLAGS):
    '''
    Generate a random tlhw
    :param FLAGS:
    :return: (top,left,height,width)
    '''
    img_shape = FLAGS.img_shape
    img_height = img_shape[0]
    img_width = img_shape[1]
    maxt = img_height - FLAGS.vertical_margin - FLAGS.height
    maxl = img_width - FLAGS.horizontal_margin - FLAGS.width
    t = int(np.random.uniform(FLAGS.vertical_margin,maxt))
    l = int(np.random.uniform(FLAGS.horizontal_margin,maxl))
    h = int(FLAGS.height)
    w = int(FLAGS.width)
    return (t,l,h,w)

def bbox2mask(FLAGS,bbox,name='mask'):
    '''
    Generate mask tensor for bbox
    :param FLAGS:
    :param bbox: tuple (top,left,height,width)
    :param name:
    :return: a tensor with shape [1,1,H,W]
    '''
    def npmask(bbox,height,width,delta_h,delta_w):
        mask = np.zeros((1,1,height,width),np.float32)
        h = np.random.randint(delta_h // 2 + 1)
        w = np.random.randint(delta_w // 2 + 1)
        mask[:,:,bbox[0]+h:bbox[0]+bbox[2]-h,bbox[1]+w:bbox[1]+bbox[3]-w] = 1.
        return mask

    img_shape = FLAGS.img_shape
    height = img_shape[0]
    width = img_shape[1]
    mask = npmask(bbox,height,width,FLAGS.max_delta_height,FLAGS.max_delta_width)

    return mask

def brush_stroke_mask(FLAGS,name='mask'):
    min_num_vertex = 4
    max_num_vertex = 12
    mean_angle = 2*math.pi/5
    angle_range = 2*math.pi/15
    min_width = 12
    max_width = 40
    def generate_mask(H,W):
        average_radius = math.sqrt(H*H+W*W) /8
        mask = Image.new('L',(W,H),0)

        for _ in range(np.random.randint(1,4)):
            num_vertex = np.random.randint(min_num_vertex,max_num_vertex)
            angle_min = mean_angle - np.random.uniform(0,angle_range)
            angle_max = mean_angle + np.random.uniform(0,angle_range)

            angles = []
            vertex = []
            for i in range(num_vertex):
                if i%2 == 0:
                    angles.append(2*math.pi - np.random.uniform(angle_min,angle_max))
                else:
                    angles.append(np.random.uniform(angle_min,angle_max))

            w,h = mask.size
            vertex.append((int(np.random.randint(0,w)),int(np.random.randint(0,h))))
            for i in range(num_vertex):
                r = np.clip(np.random.normal(loc=average_radius,scale=average_radius//2),0,2*average_radius)
                new_x = np.clip(vertex[-1][0]+r*math.cos(angles[i]),0,w)
                new_y = np.clip(vertex[-1][1]+r*math.sin(angles[i]),0,h)
                vertex.append((int(new_x),int(new_y)))

            draw = ImageDraw.Draw(mask)
            width = int(np.random.uniform(min_width,max_width))
            draw.line(vertex,fill=255,width=width)
            for v in vertex:
                draw.ellipse((v[0]-width//2,v[1]-width//2,v[0]+width//2,v[1]+width//2),fill=255)

        if np.random.normal()>0:
            mask.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.normal()>0:
            mask.transpose(Image.FLIP_TOP_BOTTOM)

        #mask.show()
        mask = np.asarray(mask,np.float32)
        mask = np.reshape(mask,(1,1,H,W))
        return mask

    img_shape = FLAGS.img_shape
    height = img_shape[0]
    width = img_shape[1]
    mask = generate_mask(height,width)
    return mask



if __name__ == '__main__':
    import argparse
    paser = argparse.ArgumentParser()
    paser.add_argument('--img_shape',default=(512,256))
    args = paser.parse_args()
    mask = brush_stroke_mask(FLAGS=args)
    print('ok')



