# LMACV: A Lightweight Multiscale ACmix–VMamba Framework for Efficient Multimodal Medical Image Fusion (Neurocomputing)




## This is the official implementation of the LMACV model proposed in the paper with Pytorch.





## Requirements：

- CUDA 11.4
- conda 4.10.1
- Python 3.8.12
- PyTorch 1.9.1
- timm 0.4.12
- tqdm
- glob
- pandas



conda create -n vmamba python=3.8;  
conda activate vmamba;  
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117;  
pip install packaging;  
pip install timm==0.4.12;  
pip install pytest chardet yacs termcolor;  
pip install submitit tensorboardX;  
pip install triton==2.0.0;  
pip install causal_conv1d==1.0.0  # causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl;  
pip install mamba_ssm==1.0.1  # mmamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl;
 

## The .whl files of causal_conv1d and mamba_ssm could be found here.  [baidu](https://pan.baidu.com/s/1Tibn8Xh4FMwj0ths8Ufazw?pwd=uu5k)



# Tips:

RGBtoYUV ： Refer to [MATR](https://github.com/tthinking/MATR)

Dataset  is  [here](https://www.med.harvard.edu/AANLIB/home.html)

The code for evaluation metrics is [here](https://github.com/liuuuuu777/ImageFusion-Evaluation)   

Downstream:   
segmentation: https://blog.csdn.net/wsLJQian/article/details/124196453 请提前参考，mask是肿瘤的标签   
detection :  YOLOv8s  


#  
Cite the paper：









# #  Acknowledgment
This project is based on Mamba ([code](https://github.com/MzeroMiko/VMamba)), MPCT ([code](https://github.com/wangzi487794504/Image-fusion))  thanks for their excellent works.



