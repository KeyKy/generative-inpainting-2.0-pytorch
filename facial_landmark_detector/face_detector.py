import cv2
import dlib
from PIL import Image,ImageDraw

def facial_landmark_detector(face_img_path,predictor_path):
    face_img = cv2.imread(face_img_path)
    img_PIL = Image.open(face_img_path)
    gray = cv2.cvtColor(face_img,cv2.COLOR_BGR2GRAY)

    detector = dlib.get_frontal_face_detector()

    predictor = dlib.shape_predictor(predictor_path)

    dets = detector(gray,1)
    shape_list = []
    for face in dets:
        shape_list.append(predictor(face_img,face))
    return img_PIL,shape_list


if __name__ == '__main__':
    img_path = './images/000003.png'
    predictor_path = './shape_predictor_68_face_landmarks.dat'
    img,shape_list = facial_landmark_detector(img_path,predictor_path)
    draw = ImageDraw.Draw(img)
    width = 3
    pt_pos_list = []
    for shape in shape_list:
        for pt in shape.parts():
            pt_pos_list.append((pt.x,pt.y))
            #cv2.circle(img,pt_pos,2,(0,255,0),1)
    #cv2.imshow('image',img)
    #cv2.waitKey(0)
    draw.line(pt_pos_list,fill=(255,255,255),width=width)
    img.show()

