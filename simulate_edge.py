import csv
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.models import vgg16

from pps import flatten_model


# ============================================================
# 配置: 模拟 Jetson Nano + WiFi 环境
# ============================================================
BANDWIDTH_MBPS = 20.0       # 论文反推约 20 Mbps
DEVICE_SLOWDOWN = 100.0     # 把 RTX5060 的毫秒级时间放大到秒级，把"设备侧计算时间"放大 100 倍，模拟 Jetson Nano 的弱算力。
SERVER_SPEEDUP = 70.0       # RTX3090 vs Jetson Nano 约 70 倍
RESULT_SIZE_MB = 0.000004   # 4KB


def head_forward(ordered, end_layer_index, x):
    """逐层执行到 end_layer_index。"""
    h = x
    for i, (name, layer) in enumerate(ordered):
        if i > end_layer_index:
            break       #"切"
        if isinstance(layer, nn.Linear) and h.dim() > 2:    #展平全连接层
            h = torch.flatten(h, 1)
        h = layer(h)
    return h


def measure_one_layer(ordered, end_layer_index, input_tensor, device, warmup=3, repeat=10):
    """测 head 到 end_layer_index 的时间（ms）和输出大小（MB）。"""
    with torch.no_grad():
        out = head_forward(ordered, end_layer_index, input_tensor)
    size_mb = out.numel() * 4 / (1024 ** 2) #元素总数*float32每个元素4字节，再字节转MB

    with torch.no_grad():
        for _ in range(warmup):     #预热
            head_forward(ordered, end_layer_index, input_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(repeat):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            head_forward(ordered, end_layer_index, input_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()    #GPU异步执行，必须同步才能准确测时间
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

    times_sorted = sorted(times)
    trimmed = times_sorted[1:-1] if len(times_sorted) > 2 else times_sorted #去掉最大最小各 1 个，取平均
    return sum(trimmed) / len(trimmed), size_mb


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model = vgg16(weights=None).to(device)
    model.eval()
    input_tensor = torch.randn(1, 3, 224, 224).to(device)   #输入是随机生成的(1, 3, 224, 224)的张量

    ordered = flatten_model(model)
    n_layers = len(ordered)
    print(f"VGG16 总层数: {n_layers}")

    # 测完整模型时间（用于估算 T_ES）
    with torch.no_grad():
        for _ in range(3):
            model(input_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    full_times = []
    with torch.no_grad():
        for _ in range(10):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(input_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            full_times.append((t1 - t0) * 1000.0)
    full_ms = sum(sorted(full_times)[1:-1]) / (len(full_times) - 2)
    print(f"完整 VGG16 本地时间: {full_ms:.2f} ms")

    # 逐层测量
    results = []
    print("\n逐层测量中...")
    for end_idx in range(n_layers):
        T_UE_ms, upload_mb = measure_one_layer(
            ordered, end_idx, input_tensor, device
        )

        # 模拟 Jetson Nano：把 RTX5060 的时间放大
        T_UE_sim_ms = T_UE_ms * DEVICE_SLOWDOWN

        # T_up
        T_up_ms = upload_mb * 8 / BANDWIDTH_MBPS * 1000 #MB 转 Mbit（1 字节 = 8 比特）

        # T_ES: 服务器跑 tail
        tail_ms = full_ms - T_UE_ms
        T_ES_ms = max(0, tail_ms) / SERVER_SPEEDUP * DEVICE_SLOWDOWN

        # T_down
        T_down_ms = RESULT_SIZE_MB * 8 / BANDWIDTH_MBPS * 1000

        # 总延迟（秒）
        L_s = (T_UE_sim_ms + T_up_ms + T_ES_ms + T_down_ms) / 1000.0

        results.append({
            "layer_idx": end_idx,
            "layer_name": ordered[end_idx][0],
            "T_UE_s": T_UE_sim_ms / 1000.0,     #从输入到这一层的执行时间ms
            "T_up_s": T_up_ms / 1000.0,
            "T_ES_s": T_ES_ms / 1000.0,
            "T_down_s": T_down_ms / 1000.0,
            "L_s": L_s,
        })

    # 存 CSV
    with open("end_to_end_latency_per_layer.csv", "w",
              newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "layer_idx", "layer_name", "T_UE_s", "T_up_s",
            "T_ES_s", "T_down_s", "L_s"
        ])
        writer.writeheader()
        writer.writerows(results)
    print("已保存: end_to_end_latency_per_layer.csv")

    # 画图
    labels = [r["layer_name"].replace("features.", "").replace("classifier.", "fc") for r in results]
    T_UE = [r["T_UE_s"] for r in results]
    T_up = [r["T_up_s"] for r in results]
    T_ES = [r["T_ES_s"] for r in results]
    T_down = [r["T_down_s"] for r in results]

    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(16, 6))

    ax.bar(x, T_UE, label="User Equipment", color="#F4A460",
           edgecolor="black", linewidth=0.3)
    ax.bar(x, T_up, bottom=T_UE, label="Upload", color="#90EE90",
           edgecolor="black", linewidth=0.3)
    bottom2 = [a + b for a, b in zip(T_UE, T_up)]
    ax.bar(x, T_ES, bottom=bottom2, label="Edge Server", color="#B0C4DE",
           edgecolor="black", linewidth=0.3)
    bottom3 = [a + b + c for a, b, c in zip(T_UE, T_up, T_ES)]
    ax.bar(x, T_down, bottom=bottom3, label="Download", color="#FFFF99",
           edgecolor="black", linewidth=0.3)        #四根柱子堆叠在一起，总高度就是端到端延迟 L

    ax.set_xlabel("Partition Point", fontsize=12)
    ax.set_ylabel("Latency (s)", fontsize=12)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("fig1_reproduction.png", dpi=150)
    print("已保存: fig1_reproduction.png")


if __name__ == "__main__":
    main()