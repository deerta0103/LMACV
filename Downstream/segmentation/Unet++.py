import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import glob
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
matplotlib.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# ---------------- UNet++ 定义 ----------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=1, out_channels=4):
        super(UNetPlusPlus, self).__init__()

        self.filters = [64, 128, 256, 512, 1024]

        # Encoder
        self.enc1 = ConvBlock(in_channels, self.filters[0])
        self.enc2 = ConvBlock(self.filters[0], self.filters[1])
        self.enc3 = ConvBlock(self.filters[1], self.filters[2])
        self.enc4 = ConvBlock(self.filters[2], self.filters[3])
        self.enc5 = ConvBlock(self.filters[3], self.filters[4])

        # Decoder
        self.up4 = nn.ConvTranspose2d(self.filters[4], self.filters[3], 2, stride=2)
        self.dec4 = ConvBlock(self.filters[4], self.filters[3])
        self.up3 = nn.ConvTranspose2d(self.filters[3], self.filters[2], 2, stride=2)
        self.dec3 = ConvBlock(self.filters[3], self.filters[2])
        self.up2 = nn.ConvTranspose2d(self.filters[2], self.filters[1], 2, stride=2)
        self.dec2 = ConvBlock(self.filters[2], self.filters[1])
        self.up1 = nn.ConvTranspose2d(self.filters[1], self.filters[0], 2, stride=2)
        self.dec1 = ConvBlock(self.filters[1], self.filters[0])

        # Final layer
        self.final = nn.Conv2d(self.filters[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        e5 = self.enc5(F.max_pool2d(e4, 2))

        # Decoder
        d4 = self.dec4(torch.cat([self.up4(e5), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.final(d1)

# ---------------- 数据集定义 ----------------
class BrainTumorDataset(Dataset):
    """
    标签使用整数形式：0=背景，1=绿色，2=黄色，3=红色
    输入图像：灰度图像
    标签图：每像素值为 0/1/2/3
    """
    def __init__(self, img_paths, mask_paths):
        self.img_paths = img_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        # 归一化图像，转为 [1, H, W]
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        # 标签映射：原始像素值 → 类别编号
        label = np.zeros_like(mask, dtype=np.uint8)
        label[mask == 127] = 1
        label[mask == 255] = 2
        label[mask == 63]  = 3
        label[mask == 0]   = 0

        return torch.from_numpy(img).float(), torch.from_numpy(label).long()

# ---------------- 训练函数 ----------------
def train_multiclass_unet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    img_paths = sorted(glob.glob('F:\DaBaiCai\yuanshisigexulie\shaoxuan\DFEnet\*.png'))
    mask_paths = sorted(glob.glob('F:\DaBaiCai\yuanshisigexulie\shaoxuan\seg\*.png'))

    dataset = BrainTumorDataset(img_paths, mask_paths)
    loader = DataLoader(dataset, batch_size=8, num_workers=6, shuffle=True)

    model = UNetPlusPlus(in_channels=1, out_channels=4).to(device)

    # 使用类别权重
    weights = torch.tensor([0.2, 0.8, 3.0, 1.0], device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print("开始训练...")
    from tqdm import tqdm
    for epoch in range(800):
        model.train()
        total_loss = 0

        print(f"\n[Epoch {epoch + 1}] 开始训练")
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}", unit="batch")

        for img, mask in pbar:
            img, mask = img.to(device), mask.to(device)

            pred = model(img)
            loss = criterion(pred, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        print(f"[Epoch {epoch + 1}] 总Loss: {total_loss:.4f}")

        # 每个 epoch 结束时保存模型
        model_path = f'multiclass_unetpp_epoch_{epoch + 1}.pth'
        torch.save(model.state_dict(), model_path)
        print(f"模型已保存为 {model_path}")

    print("训练完成")

# ---------------- 预测+可视化函数 ----------------
def predict_multiclass_rgb(image_path, model_path='multiclass_unetpp.pth', save_path='colored_mask.png'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert("L")
    image = image.resize((256, 256))
    input_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)

    model = UNetPlusPlus(in_channels=1, out_channels=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        output = model(input_tensor)  # [1, 4, H, W]
        pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    # 调试打印类别
    print(f"预测图中包含的标签类别: {np.unique(pred_mask)}")

    # 映射颜色
    color_map = {
        0: [0, 0, 0],        # 背景
        1: [0, 255, 0],      # 绿色（外层）
        2: [255, 255, 0],    # 黄色（中间层）
        3: [255, 0, 0],      # 红色（最深）
    }

    h, w = pred_mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            rgb_mask[i, j] = color_map.get(pred_mask[i, j], [255, 255, 255])

    Image.fromarray(rgb_mask).save(save_path)
    print(f"已保存彩色预测图到：{save_path}")

    # 可视化
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("原图")
    plt.imshow(image, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title("预测标签")
    plt.imshow(rgb_mask)
    plt.show()

# ---------------- 主函数入口 ----------------
if __name__ == '__main__':
    # 第一次运行先训练（只需运行一次）
    # train_multiclass_unet()

    # 然后运行预测（可以改路径）
    predict_multiclass_rgb(
        image_path='H:/DaBaiCai/yuanshisigexulie/shaoxuan/LMVA/5.png',
        model_path='multiclass_unet2_epochlvmv_319.pth',
        save_path='./results/105_mask.png'
    )