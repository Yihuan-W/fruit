"""
Open Demo: 多模态分类网络训练（只输出训练/验证损失）
- 不依赖外部数据：直接调用 build_multiscale_dataset.get_open_demo_dataset 生成合成样本
- 不保存模型/图像/CSV，不绘制混淆矩阵或 t-SNE
- 仅展示：模型结构 + 训练/验证损失随 epoch 变化
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

# ===== 从同目录导入演示数据生成器 =====
from build_multiscale_dataset import get_open_demo_dataset, CLASSES

# ---------------- 模型定义 ----------------
class TimeDomainCNN(nn.Module):
    def __init__(self, num_classes, input_length=100):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(1, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):                # x: (B,1,L)
        f = self.backbone(x).flatten(1)  # (B,128)
        return self.head(f)

class SpectrumCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.head = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):                # x: (B,1,F,T)
        f = self.backbone(x).flatten(1)  # (B,64)
        return self.head(f)

class FusionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.time_branch = nn.Sequential(
            nn.Conv1d(1, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.spec_branch = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.head = nn.Sequential(
            nn.Linear(64+32, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, xt, xs):
        ft = self.time_branch(xt).flatten(1)
        fs = self.spec_branch(xs).flatten(1)
        return self.head(torch.cat([ft, fs], dim=1))

# ---------------- 训练与评估 ----------------
def run_train_demo(model_name="fusion", epochs=12, batch_size=64, lr=1e-3, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42); np.random.seed(42)

    # 1) 生成演示数据（不落盘）
    X_t, X_s, Y = get_open_demo_dataset(window_sec=12, total_duration=180.0, fs=12.5, samples_per_class=120)
    num_classes = len(np.unique(Y))
    print(f"数据集：N={len(Y)}, 类别数={num_classes}")

    # 2) 构建 DataLoader（8:2 划分训练/验证）
    N = len(Y)
    tds = TensorDataset(torch.from_numpy(X_t), torch.from_numpy(X_s), torch.from_numpy(Y))
    n_train = int(0.8*N)
    n_val = N - n_train
    ds_train, ds_val = random_split(tds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False)

    # 3) 选择模型
    if model_name == "time":
        model = TimeDomainCNN(num_classes, input_length=X_t.shape[-1]).to(device)
    elif model_name == "spec":
        model = SpectrumCNN(num_classes).to(device)
    else:
        model = FusionModel(num_classes).to(device)

    # 4) 优化器/损失
    criterion = nn.CrossEntropyLoss()
    optimzier = optim.Adam(model.parameters(), lr=lr)

    # 5) 训练循环：仅记录/打印 “训练损失&验证损失”
    tr_losses, va_losses = [], []
    for ep in range(1, epochs+1):
        # ---- train ----
        model.train()
        run_loss, n_batch = 0.0, 0
        for xt, xs, y in dl_train:
            xt, xs, y = xt.to(device), xs.to(device), y.to(device)
            optimzier.zero_grad(set_to_none=True)
            if isinstance(model, FusionModel):
                logits = model(xt, xs)
            elif isinstance(model, TimeDomainCNN):
                logits = model(xt)
            else:
                logits = model(xs)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimzier.step()
            run_loss += float(loss.item()); n_batch += 1
        tr_loss = run_loss / max(1, n_batch)
        tr_losses.append(tr_loss)

        # ---- validate ----
        model.eval()
        run_loss, n_batch = 0.0, 0
        with torch.no_grad():
            for xt, xs, y in dl_val:
                xt, xs, y = xt.to(device), xs.to(device), y.to(device)
                if isinstance(model, FusionModel):
                    logits = model(xt, xs)
                elif isinstance(model, TimeDomainCNN):
                    logits = model(xt)
                else:
                    logits = model(xs)
                loss = criterion(logits, y)
                run_loss += float(loss.item()); n_batch += 1
        va_loss = run_loss / max(1, n_batch)
        va_losses.append(va_loss)

        print(f"[Epoch {ep:02d}/{epochs}] TrainLoss={tr_loss:.4f}  ValLoss={va_loss:.4f}")

    print("训练完成（本演示不保存模型、不绘图、不导出结果）。")
    return tr_losses, va_losses

if __name__ == "__main__":
    # 可选：model_name ∈ {"time","spec","fusion"}
    run_train_demo(model_name="fusion", epochs=15, batch_size=64, lr=1e-3)
