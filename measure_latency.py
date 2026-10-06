import csv
import time
import torch
import torch.nn as nn
from torchvision.models import vgg16

from pps import flatten_model, table2_partition


def build_head_features_only(model, end_layer_index, ordered):
    """
    为 Layer0~Layer13 构造 head model。
    前提: end_layer_index 对应的层在 features 里。

    返回:
        nn.Sequential 或 None（None 表示全卸载）
    """
    if end_layer_index == -1:
        return None

    end_name = ordered[end_layer_index][0]

    if not end_name.startswith("features."):
        # 涉及 classifier，本函数不处理
        return None

    # features.* 在 flatten_model 里的索引 0~30 正好对应 features[0]~features[30]
    features_end = end_layer_index + 1
    head = nn.Sequential(*list(model.features[:features_end]))
    return head


def measure_head(model_head, input_tensor, device, warmup=5, repeat=20):
    """
    测 head model 的前向时间（毫秒）。
    """
    if model_head is None:
        return 0.0

    model_head = model_head.to(device)
    model_head.eval()

    # 预热
    with torch.no_grad():
        for _ in range(warmup):
            model_head(input_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # 测量
    times = []
    with torch.no_grad():
        for _ in range(repeat):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model_head(input_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

    # 去掉最大最小各 2 个，取平均
    times_sorted = sorted(times)
    trimmed = times_sorted[2:-2] if len(times_sorted) > 4 else times_sorted
    return sum(trimmed) / len(trimmed)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model = vgg16(weights=None).to(device)
    model.eval()

    input_tensor = torch.randn(1, 3, 224, 224).to(device)

    groups, ordered = table2_partition(model)
    print(f"候选点数量: {len(groups)}\n")

    results = []
    print("=" * 75)
    print(f"{'候选点':<8}{'结束层':<20}{'本地推理时间(ms)':<20}{'备注'}")
    print("-" * 75)

    for g in groups:
        gid = g["index"]
        end_idx = g["end_layer_index"]

        if end_idx == -1:
            latency_ms = 0.0
            end_name = "Input"
            note = "全卸载，本地不跑"
        else:
            end_name = ordered[end_idx][0]
            head = build_head_features_only(model, end_idx, ordered)
            if head is None:
                latency_ms = None
                note = "涉及 classifier，Day 5 处理"
            else:
                latency_ms = measure_head(head, input_tensor, device)
                note = ""

        if latency_ms is not None:
            print(f"Layer{gid:<3}{end_name:<20}{latency_ms:<20.4f}{note}")
            results.append({
                "split_point": gid,
                "end_layer_index": end_idx,
                "end_layer_name": end_name,
                "local_latency_ms": round(latency_ms, 4),
            })
        else:
            print(f"Layer{gid:<3}{end_name:<20}{'--':<20}{note}")

    # 存 CSV（只存测到的）
    csv_path = "local_latency.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split_point", "end_layer_index",
                        "end_layer_name", "local_latency_ms"]
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 75)
    print(f"结果已保存到: {csv_path}")
    print(f"共测到 {len(results)} 个候选点")
    print("=" * 75)


if __name__ == "__main__":
    main()