import torch
import torch.nn as nn
import torch.nn.functional as F


class DNN_P(nn.Module):
    """
    DNN-p：负责决定 DNN-t 的分区策略。
    结构对应论文 Table 5：
        Layer1: Linear(input_dim, 64) + ReLU       //维度3->64
        Layer2: Linear(64, 64) + ReLU              //在64维空间做非线性变换
        Layer3: Linear(64, output_dim) + Softmax   //输出维度18
    """
    def __init__(self, input_dim=3, output_dim=18):
        super().__init__()                          #初始化（父类）nn.Module 的内部结构
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, s):
        """
        s: 当前状态，shape = (batch_size, 3)，三个分量是 [R, Q, h]
        返回:
            o: logits(未归一化的分数)，shape = (batch_size, output_dim) 调试用
            p: softmax 后的概率分布，shape = (batch_size, output_dim)   采样用
        """
        x = F.relu(self.fc1(s))
        x = F.relu(self.fc2(x))
        o = self.fc3(x)                 # logits
        p = F.softmax(o, dim=-1)        #式(4) p_i = e^{o_i} / Σ_j e^{o_j}，在最后一个维度上
                                        # 即每个样本的18个分数之间做归一化，解释成概率
        return o, p

if __name__ == "__main__":
    # 单独测试 forward
    dnn_p = DNN_P(input_dim=3, output_dim=18)
    s = torch.tensor([[27.0, 1.0, -40.0]], dtype=torch.float32)    #写一个假状态测试
    o, p = dnn_p(s)
    print("logits:", o)
    print("概率分布:", p)
    print("概率和:", p.sum().item())