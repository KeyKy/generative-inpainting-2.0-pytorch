# generative-inpainting-2.0-pytorch
Unofficial implementation of DeepFill v2, ICCV2019 paper Free-Form Image Inpainting with Gated Convolution <br>

## Link
* Part of the code is derived from https://github.com/WonwoongCho/Generative-Inpainting-pytorch, especially the contextual attention module.<br>
* Spectral normalization module is derived from https://github.com/christiancosgrove/pytorch-spectral-normalization-gan <br>
* According to the author,for faces,extracting landmarks and then connecting them to generate user-guided edge maps. for natural scence, using HED edge detector to extract edge maps.In practice, I only implemented this code on [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html),using both HED and facial landmark detector to generate the edge maps of face.<br>

## usage
* check out the requirement
* download the [pretrained model](https://github.com/davisking/dlib-models) of 68-points facial landmark predictor and unzip it to `./facial_landmark_detector` folder.
* download CelebA Dataset
* preproccess the edge maps of CelebA Dataset: <br>
  `python preproccess_edge.py --face_preddictor_path xxx --img_read_path xxx --mask_save_path xxx` <br>
* run the train code after modifying `config.py`: <br>
  `python run.py`


