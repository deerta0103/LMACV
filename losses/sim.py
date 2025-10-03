import torch
import torch.nn.functional as F
from PIL import Image
from numpy import average, dot, linalg
import numpy as np
from torchvision import transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def image_similarity_vectors_via_numpy(image1, image2):
    images = [image1, image2]
    vectors = []
    norms = []
    for image in images:
        vector = []
        for pixel_tuple in image.getdata():
            vector.append(average(pixel_tuple))
        vectors.append(vector)
        # linalg=linear（线性）+algebra（代数），norm则表示范数
        # 求图片的范数
        norms.append(linalg.norm(vector, 2))
    a, b = vectors
    a_norm, b_norm = norms
    # dot返回的是点积，对二维数组（矩阵）进行计算
    res = dot(a / a_norm, b / b_norm)
    return res

def SMILoss(input1,target,batachsize):
    image1 = input1.cpu().detach()
    image1 = image1.squeeze(1)  # 压缩一维
    # print("image1=",image1.shape[0])
    image2 = target.cpu().detach()
    image2 = image2.squeeze(dim=1)  # 压缩一维
    cosin=0
    for i in range(0,image1.shape[0]):
        t_image1 = transforms.ToPILImage()(image1[i])  # 自动转换为0-255
        t_image2 = transforms.ToPILImage()(image2[i])  # 自动转换为0-255
        cosin =cosin+ image_similarity_vectors_via_numpy(t_image1, t_image2)
    return cosin/image1.shape[0]





