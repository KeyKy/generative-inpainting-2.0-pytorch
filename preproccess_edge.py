import torch
import argparse
from PIL import Image,ImageDraw
import os
from facial_landmark_detector import facial_landmark_detector
from hed_pytorch import HED_Network
import numpy as np
#preproccess hed mask for better performance of efficiency

parser = argparse.ArgumentParser()
parser.add_argument('--hed_model_path',type=str,default='./hed_pytorch/network-bsds500.pytorch')
parser.add_argument('--face_predictor_path',type=str,default='./facial_landmark_detector/shape_predictor_68_face_landmarks.dat')
parser.add_argument('--img_read_path',type=str,default='F:/pythonProgram/Dataset/CelebA/img_align_celeba_png')
parser.add_argument('--mask_save_path',type=str,default='F:/pythonProgram/Dataset/CelebA/img_align_celeba_png_edge')
parser.add_argument('--is_face',type=bool,default=True)

args = parser.parse_args()

if __name__ == '__main__':
    #img_flist = os.listdir(args.img_read_path)
    img_flist = ['%06d.png'%i for i in range(75688,202600)]
    hed_model = HED_Network()
    hed_model.load_state_dict(torch.load(args.hed_model_path))
    hed_model = hed_model.cuda().eval()

    if not os.path.exists(args.mask_save_path):
        os.mkdir(args.mask_save_path)

    for index,img_name in enumerate(img_flist):
        img = Image.open(os.path.join(args.img_read_path,img_name))

        base_name = img_name.split('.')[0]
        print(base_name)

        # HED branch
        img = np.array(img)[:,:,::-1].transpose(2,0,1).astype(np.float32) / 255.
        c, h, w = img.shape[0], img.shape[1], img.shape[2]
        img_tensor = torch.FloatTensor(img).view(1,c,h,w).cuda()
        out_tensor = hed_model(img_tensor)
        out_tensor = out_tensor[0,:,:,:].clamp(0.,1.).permute(1,2,0)[:,:,0].cpu()
        out = (out_tensor * 255.).numpy().astype(np.uint8)
        out_img = Image.fromarray(out)

        #facial landmark branch
        if args.is_face:
            _, shape_list = facial_landmark_detector(os.path.join(args.img_read_path, img_name),args.face_predictor_path)
            pt_pos_list = []
            for shape in shape_list:
                for pt in shape.parts():
                    pt_pos_list.append((pt.x,pt.y))

            if len(pt_pos_list) == 68:
                brush_width = 3

                draw = ImageDraw.Draw(out_img)
                outline = pt_pos_list[0:17]
                draw.line(outline,fill=255,width=brush_width)

                left_eyebrow = pt_pos_list[17:22]
                draw.line(left_eyebrow, fill=255, width=brush_width)

                right_eyebrow = pt_pos_list[22:27]
                draw.line(right_eyebrow, fill=255, width=brush_width)

                nose = pt_pos_list[27:36]
                nose.append(pt_pos_list[30])
                draw.line(nose, fill=255, width=brush_width)

                left_eye = pt_pos_list[36:42]
                left_eye.append(pt_pos_list[36])
                draw.line(left_eye, fill=255, width=brush_width)

                right_eye = pt_pos_list[42:48]
                right_eye.append(pt_pos_list[42])
                draw.line(right_eye, fill=255, width=brush_width)

                mouth = pt_pos_list[48:68]
                mouth.append(pt_pos_list[48])
                draw.line(mouth, fill=255, width=brush_width)


        out_name = base_name + '.png'

        out_img.save(os.path.join(args.mask_save_path,out_name))


