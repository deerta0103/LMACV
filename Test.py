import argparse
from PIL import Image
import numpy as np
import os
import torch
import time
import imageio
import torchvision.transforms as transforms
import shutil
 
from Networks.net2zhujiaxiacaiyang import MODEL as net
# python Patching.py --dataset MyDatasets/PET-MRI_ori --output_dir MyDatasets/PET-MRI_patches1 --crop_size 120
### C:\Users\Administrator\Documents\WeChat Files\wxid_km35wml8zu0q22\FileStorage\File\2024-09\MATR\MATR\models\20240920_165909_PET-MRI_patches
def parse_args():
    parser = argparse.ArgumentParser(description="Image Fusion Script")
    
    ##### spectvmamba2ceng
    parser.add_argument('--model_path', type=str,
                        default="/home/he/cx/MATRxiugai/models/SPECT2ceng/model_1.pth",
                        help='Path to the model .pth file')

    # parser.add_argument('--model_path', type=str,
    #                     default="/home/he/cx/MATRxiugai/models/spectvmamba3fenkai2cenggaixishu/model_30.pth",
    #                     help='Path to the model .pth file')

    parser.add_argument('--dataset', type=str,  default='/home/he/datasets/test/testSPECT/test')

    parser.add_argument('--output_dir', type=str, default='./Fusion_spectvmamba3fenkai3ceng111111111111111111111111111111', help='Directory to save the copied dataset and fused images')

    parser.add_argument('--gpu', default=True, help='Use GPU if available')
    parser.add_argument('--in_channel', type=int, default=2, help='Number of input channels for the model')
    return parser.parse_args()

def load_model(args):
    # 检查是否启用 GPU
    if args.gpu and torch.cuda.is_available():
        device = torch.device('cuda')  # 使用 GPU
        print("Using GPU for computations.")
    else:
        device = torch.device('cpu')  # 使用 CPU
        print("Using CPU for computations.")

    # device = torch.device('cuda:0' if args.gpu and torch.cuda.is_available() else 'cpu')
    model = net(in_channel=args.in_channel).to(device)
    # model=MODEL().cuda()
    input=torch.randn(1,2,256,256).cuda()
    output=model(input)
    print(output.shape)
    from thop import profile
    flops, params = profile(model, inputs=(input,))
    print('flops: ', flops/1e9, 'params: ', params)
    if os.path.exists(args.model_path):
        state_dict = torch.load(args.model_path, map_location=device)
        new_state_dict = {}
        for k in state_dict.keys():
            # 移除 'module.' 前缀
            if k.startswith('module.'):
                new_k = k[7:]
            else:
                new_k = k
            new_state_dict[new_k] = state_dict[k]
        # model.load_state_dict(new_state_dict)
        model.load_state_dict(state_dict, strict=False)
        print(f"Model loaded from {args.model_path}")
    else:
        raise FileNotFoundError(f"Model path {args.model_path} does not exist.")

    return model, device


    # if os.path.exists(args.model_path):
    #     state_dict = torch.load(args.model_path, map_location=device)
    #     model.load_state_dict(state_dict)
    #     print(f"Model loaded from {args.model_path}")
    # else:
    #     raise FileNotFoundError(f"Model path {args.model_path} does not exist.")
    #
    # return model, device

def is_grayscale(image):
    """检查图像是否为灰度图，RGB模式下检查所有通道的值是否相同。"""
    if image.mode == 'RGB':
        np_image = np.array(image)
        return np.all(np_image[:, :, 0] == np_image[:, :, 1]) and np.all(np_image[:, :, 1] == np_image[:, :, 2])
    return image.mode == 'L'

def determine_image_types(dataset):
    subfolders = [os.path.join(dataset, f) for f in os.listdir(dataset) if os.path.isdir(os.path.join(dataset, f))]
    if len(subfolders) != 2:
        raise ValueError("Dataset directory must contain exactly two subfolders: one for RGB and one for grayscale images.")

    # 检查两个文件夹中的第一张图片，判断哪个文件夹是 RGB，哪个是灰度图
    first_image_1 = Image.open(os.path.join(subfolders[0], os.listdir(subfolders[0])[0]))
    first_image_2 = Image.open(os.path.join(subfolders[1], os.listdir(subfolders[1])[0]))

    if is_grayscale(first_image_1) and first_image_2.mode == 'RGB':
        gray_folder = subfolders[0]
        rgb_folder = subfolders[1]
    elif first_image_1.mode == 'RGB' and is_grayscale(first_image_2):
        gray_folder = subfolders[1]
        rgb_folder = subfolders[0]
    else:
        raise ValueError("One folder must contain RGB images and the other must contain grayscale images.")

    return rgb_folder, gray_folder

def get_epoch_from_model_path(model_path):
    """Extract the epoch number from the model filename."""
    model_filename = os.path.basename(model_path)
    # Assuming the model file is named like 'model_X.pth', extract 'X'
    epoch = model_filename.split('_')[-1].split('.')[0]  # Extracts the number 'X'
    return epoch

def copy_dataset_structure(rgb_folder, gray_folder, output_dir, model_epoch):
    """Copy dataset structure and add model_epoch to the output directory name."""
    output_rgb_dir = os.path.join(output_dir, os.path.basename(rgb_folder))
    output_gray_dir = os.path.join(output_dir, os.path.basename(gray_folder))
    output_fusion_dir = os.path.join(output_dir, 'fusion')

    # 删除已存在的目标文件夹
    if os.path.exists(output_rgb_dir):
        shutil.rmtree(output_rgb_dir)
    if os.path.exists(output_gray_dir):
        shutil.rmtree(output_gray_dir)
    if os.path.exists(output_fusion_dir):
        shutil.rmtree(output_fusion_dir)

    # 复制原始 RGB 和灰度文件夹
    shutil.copytree(rgb_folder, output_rgb_dir)
    shutil.copytree(gray_folder, output_gray_dir)

    # 创建融合文件夹
    os.makedirs(output_fusion_dir, exist_ok=True)

    return output_rgb_dir, output_gray_dir, output_fusion_dir

def fusion(args, model, device):
    # Determine image types
    rgb_folder, gray_folder = determine_image_types(args.dataset)
    
    # Extract epoch number from the model filename
    model_epoch = get_epoch_from_model_path(args.model_path)

    # Modify output_dir to include epoch info
    output_dir = os.path.join("fusion_result", args.output_dir + f"_model_{model_epoch}")
    
    # Copy dataset structure and create output directories
    output_rgb_dir, output_gray_dir, output_fusion_dir = copy_dataset_structure(rgb_folder, gray_folder, output_dir, model_epoch)

    rgb_images = sorted(os.listdir(rgb_folder))
    gray_images = sorted(os.listdir(gray_folder))

    if len(rgb_images) != len(gray_images):
        raise ValueError("The number of RGB and grayscale images must be the same.")

    tran = transforms.ToTensor()

    for num, (rgb_img_name, gray_img_name) in enumerate(zip(rgb_images, gray_images)):
        tic = time.time()

        # Load images
        rgb_img_path = os.path.join(rgb_folder, rgb_img_name)
        gray_img_path = os.path.join(gray_folder, gray_img_name)

        img1 = Image.open(rgb_img_path).convert('RGB')
        img2 = Image.open(gray_img_path).convert('L')

        img1_np = np.array(img1)  # Convert img1 to NumPy array
        img1_yuv = rgb2yuv(img1_np)  # Convert RGB to YUV
        img1_org = img1_yuv[:, :, 0]  # Extract Y channel

        img2_org = img2

        img1_org = tran(img1_org)
        img2_org = tran(img2_org)

        input_img = torch.cat((img1_org, img2_org), 0).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            out = model(input_img)

        result = np.squeeze(out.cpu().numpy())
        img1_yuv[:, :, 0] = result  # Replace Y channel with the model output

        result_rgb = yuv2rgb(img1_yuv)  # Convert back to RGB

        # Use the original image names in the output filename
        output_fusion_img = os.path.join(output_fusion_dir, f'{os.path.splitext(gray_img_name)[0]}.png')
        imageio.imwrite(output_fusion_img, result_rgb)

        toc = time.time()
        print(f'Fusion {num} completed in {toc - tic:.2f} seconds. Saved at {output_fusion_img}')

def rgb2yuv(rgb):
    rgb = rgb.astype(np.float32) / 255.0
    m = np.array([[0.299, 0.587, 0.114], 
                  [-0.147, -0.289, 0.436], 
                  [0.615, -0.515, -0.100]], dtype=np.float32)
    yuv = np.dot(rgb, m.T)
    return yuv

def yuv2rgb(yuv):
    yuv = yuv.astype(np.float32)
    m = [[1.0, 0.0, 1.13983], [1.0, -0.39465, -0.58060], [1.0, 2.03211, 0.0]]
    rgb = np.empty_like(yuv)
    for i in range(3):
        rgb[:, :, i] = yuv[:, :, 0] * m[i][0] + yuv[:, :, 1] * m[i][1] + yuv[:, :, 2] * m[i][2]
    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return rgb

if __name__ == '__main__':
    args = parse_args()
    model, device = load_model(args)
    fusion(args, model, device)
