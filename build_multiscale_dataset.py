import os
import numpy as np
import pandas as pd
from scipy.signal import stft
from sklearn.model_selection import train_test_split

# ==================== 参数配置 ====================
SRC_DIR = "./fruit"     # 原始 CSV 文件路径
OUT_DIR = "./dataset"      # 输出路径
os.makedirs(OUT_DIR, exist_ok=True)

# 类别名称（文件名中必须包含这些词）
CLASSES = ["哈密瓜","松果","枣","柠檬","树莓","核桃","牛油果","玉米","草莓","荔枝"]

# 四种滑窗长度（秒）
WINDOW_SET = [6, 8, 12, 16]
STEP_SEC = 0.5             # 滑动步长（秒）可改为0.5
SPEC_NPERSEG = 128
SPEC_NOVERLAP = 64

# ==================== 工具函数 ====================
def read_csv_auto(path):
    """自动读取CSV，仅保留前两列：时间和信号"""
    for enc in ["utf-8", "gbk", "gb2312"]:
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except Exception:
            continue
    df = df.dropna(axis=0)
    # --- 自动检测列数并取前两列 ---
    if df.shape[1] >= 2:
        df = df.iloc[:, :2]
    else:
        raise ValueError(f"❌ 文件 {path} 列数不足2，无法提取时间和信号")
    df.columns = ["时间", "信号"]
    df["时间"] = pd.to_numeric(df["时间"], errors="coerce")
    df["信号"] = pd.to_numeric(df["信号"], errors="coerce")
    df = df.dropna()
    return df


def compute_sampling_rate(t):
    """由时间列计算采样率"""
    dt = np.median(np.diff(t))
    fs = 1.0 / dt
    return fs

def normalize_signal(x):
    """零均值单位方差归一化"""
    return (x - np.mean(x)) / (np.std(x) + 1e-6)

def extract_segments(sig, fs, window_sec, step_sec):
    """滑动窗口切片"""
    win_len = int(window_sec * fs)
    step_len = int(step_sec * fs)
    segs = []
    for start in range(0, len(sig) - win_len, step_len):
        segs.append(sig[start:start + win_len])
    return np.array(segs)


def compute_stft_features(sig, fs):
    """计算STFT幅度谱（弱信号增强）"""
    L = len(sig)
    nperseg = min(128, L // 2)  # 自适应窗口
    noverlap = nperseg // 2

    f, t, Zxx = stft(sig, fs=fs, nperseg=nperseg, noverlap=noverlap)
    S = np.abs(Zxx)
    S = np.log1p(S)  # 对数幅值压缩

    # 统一尺寸
    F, T = S.shape
    F_fix, T_fix = 50, 50
    S_pad = np.zeros((F_fix, T_fix))
    S_pad[:min(F, F_fix), :min(T, T_fix)] = S[:min(F, F_fix), :min(T, T_fix)]
    return S_pad


# ==================== 主函数 ====================
if __name__ == "__main__":
    for W in WINDOW_SET:
        all_Xt, all_Xs, all_Y = [], [], []
        print(f"\n==================== 🧩 窗口 {W}s ====================")

        for label, cls in enumerate(CLASSES):
            files = [f for f in os.listdir(SRC_DIR) if cls in f and f.endswith(".csv")]
            if not files:
                print(f"⚠️ 未找到类别 {cls} 文件")
                continue

            for f in files:
                path = os.path.join(SRC_DIR, f)
                df = read_csv_auto(path)
                t = df["时间"].values
                x = df["信号"].values
                fs = compute_sampling_rate(t)
                sig = normalize_signal(x)

                print(f"✅ 读取 {cls}: {len(sig)} 点 ≈ {len(sig)/fs:.1f}s, 采样率≈{fs:.2f} Hz")

                segments = extract_segments(sig, fs, W, STEP_SEC)
                for seg in segments:
                    xt = seg[np.newaxis, :]             # (1, L)
                    xs = compute_stft_features(seg, fs)[np.newaxis, :, :]  # (1, F, T)
                    all_Xt.append(xt)
                    all_Xs.append(xs)
                    all_Y.append(label)

        X_time = np.array(all_Xt)
        X_spec = np.array(all_Xs)
        Y = np.array(all_Y)
        print(f"✅ 总样本: {len(Y)}, 时域形状: {X_time.shape}, 频谱形状: {X_spec.shape}")

        # ===== 全局归一化 =====
        X_time = (X_time - X_time.mean()) / (X_time.std() + 1e-6)
        X_spec = (X_spec - X_spec.mean()) / (X_spec.std() + 1e-6)

        # ===== 分层划分训练/验证 =====
        Xtr_t, Xv_t, Xtr_s, Xv_s, Ytr, Yv = train_test_split(
            X_time, X_spec, Y, test_size=0.1, stratify=Y, random_state=42
        )

        np.savez(os.path.join(OUT_DIR, f"dataset_train_{W}s.npz"),
                 X_time=Xtr_t, X_spec=Xtr_s, Y=Ytr)
        np.savez(os.path.join(OUT_DIR, f"dataset_val_{W}s.npz"),
                 X_time=Xv_t, X_spec=Xv_s, Y=Yv)

        print(f"💾 已保存: dataset_train_{W}s.npz / dataset_val_{W}s.npz")
