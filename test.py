import torch
import random
import matplotlib.pyplot as plt

# 1) 导入你自己写的 Dataset 和 模型
from data_loader import MmwDataset  # 假设你在 data_loader.py 里定义的类是 MmwDataset
from model import FNO               # 假设你在 model.py 里定义的模型类叫 FNO

# 如果还要用 DataLoader
from torch.utils.data import DataLoader

def main():
    # ====== 1) 加载模型权重 ======
    ckpt_path = "./lightning_logs/version_0/checkpoints/epoch=899-step=8100.ckpt"
    model = FNO.load_from_checkpoint(ckpt_path)  # PyTorch Lightning的便捷接口
    model.eval()
    model.to('cpu')  # 如果有GPU可用的话，改成  model.to('cuda')

    # ====== 2) 构建 Dataset 和 DataLoader ======
    # 这里举例：假设  data_loader.py 里有 MmwDataset(root_dir, ...)，负责加载 .mat 文件等
    test_dataset = MmwDataset(
        root_dir = r"D:\Project\python\inversion\32to64\32to64DataSet",
        index_list=list(range(1101, 1211))
    )

    # 如果你想随机抽取一条样本，也可以不一定用 DataLoader；
    # 但演示一下用 DataLoader 的方式拿一批数据:
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,    # 一次只取1条
        shuffle=True,    # 让它在迭代时随机
        num_workers=0
    )

    # 取一条数据 (batch)
    # 方式 A: 直接 next(iter(...))
    batch = next(iter(test_loader))  # 这会返回 (input_data, target_img)
    input_data, target_img = batch

    # 方式 B: 也可以手动随机 index => test_dataset[idx]
    # idx = random.randint(0, len(test_dataset) - 1)
    # input_data, target_img = test_dataset[idx]
    # # 加 batch 维 => unsqueeze(0)
    # input_data = input_data.unsqueeze(0)
    # target_img = target_img.unsqueeze(0)

    # 打印一下形状
    print("Input shape :", input_data.shape)   # 期望 [batch=1, 2, 32, 32] 或 [1, C, H, W]
    print("Target shape:", target_img.shape)   # 期望 [1, 1, 32, 32]

    # ====== 3) 推理 ======
    # 如果你在 CPU 上推理：.to('cpu')；如果在 GPU 上：.to('cuda')
    input_data  = input_data.to(model.device)
    target_img  = target_img.to(model.device)

    with torch.no_grad():
        pred = model(input_data)  # => [batch=1, 1, 32, 32] （以你代码实际输出为准）

    # ====== 4) 可视化对比 ======
    # 取第一张 (其实就只有一张)
    pred_img   = pred[0].detach().cpu().numpy()    # => [1, 32, 32]
    groundtruth = target_img[0].detach().cpu().numpy()

    # squeeze 掉通道维 => [32, 32]
    pred_img   = pred_img.squeeze()
    groundtruth = groundtruth.squeeze()

    # matplotlib 显示
    fig, axes = plt.subplots(1, 2, figsize=(6,3))
    axes[0].imshow(pred_img, cmap='gray')
    axes[0].set_title("Predicted")

    axes[1].imshow(groundtruth, cmap='gray')
    axes[1].set_title("Ground Truth")

    plt.show()

if __name__ == "__main__":
    main()