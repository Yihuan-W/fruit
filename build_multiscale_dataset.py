"""
Open Demo: 多尺度数据管线（不读CSV，不落盘）
- 用可控的“合成信号”替代真实CSV，演示：滑窗切片、归一化、STFT特征。
- 仅打印数据形状和流程说明，不保存 .npz 文件。
"""

import numpy as np
from scipy.signal import stft

# ============== 可配置参数（仅用于演示） ==============
CLASSES = ["哈密瓜","松果","枣","柠檬","树莓","核桃","牛油果","玉米","草莓","荔枝"]
WINDOW_SET = [6, 8, 12, 16]   # 多种窗口时长（秒）
STEP_SEC   = 0.5              # 滑动步长（秒）
F_FIX, T_FIX = 50, 50         # 统一后的谱图大小
SEED = 42

# ============== 合成信号：模拟不同类别的“振动特征” ==============
def _make_sine_mixture(duration_s: float, fs: float, base_freq: float, drift: float, noise: float):
    """
    生成：若干余弦 + 缓慢频率漂移 + 高斯噪声
    """
    t = np.arange(0, duration_s, 1.0/fs, dtype=np.float32)
    f1 = base_freq * (1.0 + 0.05*np.sin(2*np.pi*drift*t))
    f2 = 1.7*base_freq
    sig = (np.cos(2*np.pi*f1*t) + 0.6*np.cos(2*np.pi*f2*t + 0.3)) * (0.5+0.5*np.cos(2*np.pi*0.1*t))
    sig += noise*np.random.randn(len(t)).astype(np.float32)
    return t, sig

def _normalize(x):
    return (x - x.mean()) / (x.std() + 1e-6)

def _extract_segments(sig, fs, window_sec, step_sec):
    win_len  = int(window_sec * fs)
    step_len = int(step_sec * fs)
    segs = []
    for s in range(0, max(0, len(sig)-win_len), step_len):
        segs.append(sig[s:s+win_len])
    return np.array(segs, dtype=np.float32)

def _stft_pad(sig, fs, f_fix=F_FIX, t_fix=T_FIX):
    # 自适应窗口
    L = len(sig)
    nperseg = max(16, min(128, L//4))
    noverlap = nperseg//2
    f, tt, Z = stft(sig, fs=fs, nperseg=nperseg, noverlap=noverlap)
    S = np.log1p(np.abs(Z)).astype(np.float32)
    F, T = S.shape
    out = np.zeros((f_fix, t_fix), dtype=np.float32)
    out[:min(F,f_fix), :min(T,t_fix)] = S[:min(F,f_fix), :min(T,t_fix)]
    return out

def get_open_demo_dataset(window_sec=12, total_duration=120.0, fs=12.5,
                          samples_per_class=80, rng=np.random.RandomState(SEED)):
    """
    生成“演示集”：只返回内存中的 (X_time, X_spec, Y)，不落盘。
    - 每个类别先生成一条长序列（可选不同 pattern），再用滑窗切片成样本。
    """
    all_xt, all_xs, all_y = [], [], []

    # 为每个类别定义不同的基础频率/漂移/噪声，体现“类差异”
    base_bank = np.linspace(0.2, 2.0, num=len(CLASSES))   # Hz
    for label, cls in enumerate(CLASSES):
        base = base_bank[label]
        drift = 0.03 + 0.02*(label % 3 + 1)
        noise = 0.15 + 0.03*(label % 4)

        # 生成一条较长时间的“原始记录”
        t, raw = _make_sine_mixture(duration_s=total_duration, fs=fs,
                                    base_freq=base, drift=drift, noise=noise)
        raw = _normalize(raw)

        # 滑窗切片
        segs = _extract_segments(raw, fs, window_sec=window_sec, step_sec=STEP_SEC)
        if len(segs) == 0:
            continue

        # 随机抽样固定数量的样本（演示）
        idx = rng.choice(len(segs), size=min(samples_per_class, len(segs)), replace=False)
        segs = segs[idx]

        # 计算时域/频域特征
        for seg in segs:
            xt = seg[np.newaxis, :]                       # (1, L)
            xs = _stft_pad(seg, fs)[np.newaxis, :, :]     # (1, F, T)
            all_xt.append(xt); all_xs.append(xs); all_y.append(label)

    X_time = np.array(all_xt, dtype=np.float32)
    X_spec = np.array(all_xs, dtype=np.float32)
    Y      = np.array(all_y,  dtype=np.int64)

    # 全局归一化（演示）
    X_time = (X_time - X_time.mean()) / (X_time.std() + 1e-6)
    X_spec = (X_spec - X_spec.mean()) / (X_spec.std() + 1e-6)

    return X_time, X_spec, Y

if __name__ == "__main__":
    np.random.seed(SEED)
    print("🔧 以 12s 窗口演示多模态数据管线（合成信号）……")
    X_t, X_s, Y = get_open_demo_dataset(window_sec=12, total_duration=180.0, fs=12.5, samples_per_class=100)
    print(f"✅ 样本数: {len(Y)}")
    print(f"✅ 时域形状: {X_t.shape}  (N, 1, L)")
    print(f"✅ 频域形状: {X_s.shape}  (N, 1, {F_FIX}, {T_FIX})")
    print("📌 说明：此脚本仅展示数据处理流程，不保存任何文件。")
