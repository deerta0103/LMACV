import os
import argparse
from PIL import Image
import numpy as np
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description='Overlapping Crop Data Augmentation')
    # parser.add_argument('--dataset', type=str, required=True, default='MyDatasets/PET-MRI_ori',
    #                     help='Path to the dataset containing train and test folders')
    #
    # parser.add_argument('--output_dir', type=str, required=True, default="MyDatasets/PET-MRI_patches1",
    #                     help='Directory to save the new dataset with cropped patches')
    parser.add_argument('--dataset', type=str,  default='MyDatasets/PET-MRI_ori',
                        help='Path to the dataset containing train and test folders')

    parser.add_argument('--output_dir', type=str,  default="MyDatasets/PET-MRI_patches1",
                        help='Directory to save the new dataset with cropped patches')



    parser.add_argument('--crop_size', type=int, default=120, 
                        help='Size of the cropped patches (default: 120)')
    parser.add_argument('--stride', type=int, default=20, 
                        help='Stride for the overlapping crops (default: 20)')
    args = parser.parse_args()
    return args

def get_modalities(train_dir):
    # 获取train目录下的两个模态子文件夹
    subfolders = [f.path for f in os.scandir(train_dir) if f.is_dir()]
    if len(subfolders) != 2:
        raise ValueError(f"Expected 2 modality folders in {train_dir}, but found {len(subfolders)}.")
    
    return subfolders[0], subfolders[1]

def create_output_dirs(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    train_output_dir = os.path.join(output_dir, 'train')
    test_output_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)
    return train_output_dir, test_output_dir

def crop_image(image, crop_size, stride):
    width, height = image.size
    cropped_patches = []
    for i in range(0, width - crop_size + 1, stride):
        for j in range(0, height - crop_size + 1, stride):
            patch = image.crop((i, j, i + crop_size, j + crop_size))
            cropped_patches.append(patch)
    return cropped_patches

def process_and_save_cropped(modal_dir, output_dir, crop_size, stride):
    filenames = [f for f in os.listdir(modal_dir) if f.endswith('.bmp') or f.endswith('.png') or f.endswith('.jpg')]
    for filename in filenames:
        img_path = os.path.join(modal_dir, filename)
        img = Image.open(img_path)
        patches = crop_image(img, crop_size, stride)

        base_name = os.path.splitext(filename)[0]
        for idx, patch in enumerate(patches):
            patch.save(os.path.join(output_dir, f"{base_name}_patch_{idx}.png"))

def copy_non_cropped_data(src_dir, dest_dir):
    # 如果目标目录已经存在，先删除它
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)  # 删除已有目录及其内容
    # 复制目录
    shutil.copytree(src_dir, dest_dir)

def main():
    args = parse_args()

    # 确定train和test的路径
    train_dir = os.path.join(args.dataset, 'train')
    test_dir = os.path.join(args.dataset, 'test')

    # 找到两个模态文件夹路径
    modality_1, modality_2 = get_modalities(train_dir)
    
    # 创建输出目录，并在其中创建train和test的子目录
    train_output_dir, test_output_dir = create_output_dirs(args.output_dir)

    # 在train目录下为两个模态创建输出子目录
    modality_1_name = os.path.basename(modality_1)
    modality_2_name = os.path.basename(modality_2)
    modality_1_output_dir = os.path.join(train_output_dir, modality_1_name)
    modality_2_output_dir = os.path.join(train_output_dir, modality_2_name)
    os.makedirs(modality_1_output_dir, exist_ok=True)
    os.makedirs(modality_2_output_dir, exist_ok=True)

    # 裁剪train目录中的两个模态
    print(f"Processing {modality_1_name} dataset...")
    process_and_save_cropped(modality_1, modality_1_output_dir, args.crop_size, args.stride)
    print(f"Processing {modality_2_name} dataset...")
    process_and_save_cropped(modality_2, modality_2_output_dir, args.crop_size, args.stride)

    # 将test目录拷贝到新的数据集路径中
    print(f"Copying test data to {test_output_dir}...")
    copy_non_cropped_data(test_dir, os.path.join(test_output_dir, ''))

    print(f"Data augmentation with overlapping crops completed. New dataset saved in {args.output_dir}.")

if __name__ == '__main__':
    main()