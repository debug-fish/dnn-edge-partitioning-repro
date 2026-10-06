import torch
import torch.nn as nn
from torchvision.models import vgg16

TABLE2_END_INDICES = [                              #直接从Table_2反推
    -1,   # Layer0: Input-0
     1,   # Layer1: Conv2d-1, Relu-2
     4,   # Layer2: Conv2d-3, Relu-4, MaxPool2d-5
     6,   # Layer3: Conv2d-6, Relu-7
     9,   # Layer4: Conv2d-8, Relu-9, MaxPool2d-10
    11,   # Layer5: Conv2d-11, Relu-12
    13,   # Layer6: Conv2d-13, Relu-14
    16,   # Layer7: Conv2d-15, Relu-16, MaxPool2d-17
    18,   # Layer8: Conv2d-18, Relu-19
    20,   # Layer9: Conv2d-20, Relu-21
    23,   # Layer10: Conv2d-22, Relu-23, MaxPool2d-24
    25,   # Layer11: Conv2d-25, Relu-26
    27,   # Layer12: Conv2d-27, Relu-28
    30,   # Layer13: Conv2d-29, Relu-30, MaxPool2d-31
    31,   # Layer14: AvgPool2d-32
    34,   # Layer15: Linear-33, Relu-34, Dropout-35
    37,   # Layer16: Linear-36, Relu-37, Dropout-38
    38,   # Layer17: Linear-39
]

def flatten_model(model):
    ordered = []
    for name, layer in model.named_modules():
        if layer is model or isinstance(layer, nn.Sequential):  #跳过VGG本身和容器（过滤）
            continue
        ordered.append((name, layer))
    return ordered


def table2_partition(model):
    ordered = flatten_model(model)
    groups = []
    prev_end = -1           #记录上一个候选点的结束索引，初始化为 -1（表示还没有候选点）
    for gid, end_idx in enumerate(TABLE2_END_INDICES):
        if gid == 0:        #Layer0单独处理
            groups.append({
                "index": 0,
                "layers": ["Input-0"],
                "end_layer_index": -1,
            })
            continue
        start = prev_end + 1
        end = end_idx + 1
        group_layers = [ordered[i][0] for i in range(start, min(end, len(ordered)))]    #加min防止越界
        groups.append({
            "index": gid,
            "layers": group_layers,
            "end_layer_index": end_idx,
        })
        prev_end = end_idx
    return groups, ordered


if __name__ == "__main__":
    model = vgg16(weights=None)     #加载vgg16模型结构，不加载预训练权重
    model.eval()                    #评估模式
    ordered = flatten_model(model)

    print("=" * 80)
    print("VGG16 展开后的所有具体层")
    print("=" * 80)

    for i, (name, layer) in enumerate(ordered):
        print(f"[{i:2d}] {name:<22} {type(layer).__name__}")
    print(f"\n具体层总数: {len(ordered)}")

    print("\n" + "=" * 80)
    print("Table 2 对齐后的 18 个候选点")
    print("=" * 80)

    groups, _ = table2_partition(model)
    for g in groups:
        layers_str = " + ".join(g["layers"])
        print(f"Layer{g['index']:<3}{layers_str}")

    print("\n" + "=" * 80)
    print(f"候选点数量: {len(groups)}")
    
    if len(groups) == 18:
        print("✓ 与论文 Table 2 的 18 个候选点一致")
    else:
        print(f"✗ 期望 18，实际 {len(groups)}")
    print("=" * 80)