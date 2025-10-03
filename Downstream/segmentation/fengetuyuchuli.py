import os
from PIL import Image
import numpy as np

input_dir = "./raw_masks"
output_dir = "./masks"
os.makedirs(output_dir, exist_ok=True)

def convert_mask_to_labels(image):
    mask = np.array(image.convert("L"))  # 强制转灰度
    label_mask = np.zeros_like(mask, dtype=np.uint8)

    label_mask[mask == 0] = 0     # 背景（黑色）
    label_mask[mask == 255] = 1   # 绿色 → 最外层
    label_mask[mask == 127] = 2   # 黄色 → 中间层
    label_mask[mask == 63] = 3    # 红色 → 最里面

    return Image.fromarray(label_mask)

for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(input_dir, filename)
        save_path = os.path.join(output_dir, filename)

        image = Image.open(img_path)
        label = convert_mask_to_labels(image)
        label.save(save_path)
        print(f"已转换: {filename} → 标签图")

print("✅ 所有掩码图转换完成。")
