import os

# 指定你的图片文件夹路径
directory = r'F:\AAAlunwwwww\data\Fusion_metric_python\SPECTdataset\xinxinbnrelu\fusionxin'


directory = r'F:\AAAlunwwwww\data\Fusion_metric_python\PETdataset\changshi'

directory = r'/home/he/cx/MATRxiugai/fusion_result/Fusion_netyuanshi_model_21/fusion'
# 指定你想要从每个文件名前去掉的字符数
chars_to_remove = 6


# 检查文件扩展名是否是图片格式的函数
def is_image_file(filename):
    return any(filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp'])


# 遍历目录中的文件
for filename in os.listdir(directory):
    if is_image_file(filename):
        # 生成新的文件名
        new_filename = filename[chars_to_remove:]
        # 获取旧文件的完整路径和新文件的完整路径
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_filename)

        # 重命名文件
        os.rename(old_file, new_file)
        print(f'Renamed "{filename}" to "{new_filename}"')

print('Finished renaming files.')