import os
import cv2  # OpenCV库，用于图像处理

def convert_rgb_to_y_component(input_folder, output_folder):
    # 如果输出文件夹存在，则清空
    if os.path.exists(output_folder):
        for file_name in os.listdir(output_folder):
            file_path = os.path.join(output_folder, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    else:
        os.makedirs(output_folder)

    # 获取输入文件夹中所有的 .png 文件
    files = [f for f in os.listdir(input_folder) if
             f.endswith('.png') and os.path.isfile(os.path.join(input_folder, f))]

    for file_name in files:
        # 构造完整的输入文件路径
        input_file_path = os.path.join(input_folder, file_name)

        # 读取图片
        image = cv2.imread(input_file_path)

        # 检查图片是否成功读取
        if image is None:
            print(f"Error reading {file_name}. Skipping this file.")
            continue

        # 转换为 YUV 色彩空间
        yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

        # 提取 Y 分量
        y_component = yuv_image[:, :, 0]

        # 构造新的文件名，添加后缀 "_Y"
        new_file_name = os.path.splitext(file_name)[0] + '.png'

        # 构造完整的输出文件路径
        output_file_path = os.path.join(output_folder, new_file_name)

        # 保存 Y 分量图像到输出文件夹
        cv2.imwrite(output_file_path, y_component)
        print(f'Saved Y component of {file_name} as {new_file_name} in {output_folder}')

# 输入和输出文件夹路径
input_folder = '/home/he/cx/MATRxiugai/fusion_result/Fusion_net3248511_model_40/fusion'
output_folder = "/home/he/cx/MATRxiugai/fusion_result/Fusion_net3248511_model_40/Yfusion"



# /home/he/cx/MATRxiugai/fusion_result/Fusion_netyuanshi_model_21/fusion



convert_rgb_to_y_component(input_folder, output_folder)