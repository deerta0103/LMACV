from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from color import RGB_YUV as yuv
torch.set_printoptions(threshold=np.inf)
np.set_printoptions(threshold=np.inf)
import cv2
import matplotlib.image as mplimg
# a=Image.open("train_image/mri/1.bmp").convert('L')
# b=Image.open("source images/PET-MRI/PET.png").convert('YCbCr')
# # b.save("test.bmp")
# print(b)
# b=np.array(b)
#
# b = b[:, :, ::-1]
# print(np.array(a).shape)
# # b=transforms.ToTensor()(b)
# print(b.shape)
# mplimg.imsave("test3.png",b)
# print(b)
# a=np.array(a)
# change=yuv()
# e,d,f,g=change.rgb_to_ycbcr(b)
# print(g.shape)
# n=change.ycbcr_to_rgb(im_ycbcr=g)
# print(n.shape)
# RES=Image.fromarray(n).convert("RGB")
#
# b=Image.fromarray(b)
# b.save("test2.bmp")
# RES.save("color.bmp")
# # cv2.imwrite("color.bmp",d)
img2=cv2.imread(r"train_image/mri/1.bmp")
img=cv2.imread(r"source images/PET-MRI/PET.png")
# img=img[...,::-1]
print("img=",img)
# change=yuv()
# a,b,c,d=change.rgb_to_ycbcr(im_rgb=img)
# img=change.ycbcr_to_rgb(im_ycbcr=d)
img=cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
y,u,v=cv2.split(img)
img = cv2.merge([img2,u,v])
img=cv2.cvtColor(img,cv2.COLOR_YCrCb2BGR)
# img=img[...,::-1]

cv2.imwrite("test3.png",img)
print("aa",img)

