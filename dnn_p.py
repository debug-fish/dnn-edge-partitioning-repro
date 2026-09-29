import torch
import torch.nn as nn
import torch.nn.functional as F


class DNN_P(nn.Module):
    """
    DNN-p：负责决定 DNN-t 的分区策略。
    结构对应论文 Table 5：
        Layer1: Linear(input_dim, 64) + ReLU
        Layer2: Linear(64, 64) + ReLU
        Layer3: Linear(64, output_dim) + Softmax
    """
    def __init__(self, input_dim=3, output_dim=18):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, s):
        """
        s: 当前状态，shape = (batch_size, 3)，三个分量是 [R, Q, h]
        返回:
            o: logits，shape = (batch_size, output_dim)
            p: softmax 后的概率分布，shape = (batch_size, output_dim)
        """
        x = F.relu(self.fc1(s))
        x = F.relu(self.fc2(x))
        o = self.fc3(x)                 # logits
        p = F.softmax(o, dim=-1)        # 式(4)
        return o, p

if __name__ == "__main__":
    # 单独测试 forward
    dnn_p = DNN_P(input_dim=3, output_dim=18)
    s = torch.tensor([[27.0, 1.0, -40.0]], dtype=torch.float32)
    o, p = dnn_p(s)
    print("logits:", o)
    print("概率分布:", p)
    print("概率和:", p.sum().item())