import torch
import numpy as np
import matplotlib.pyplot as plt

from dnn_p import DNN_P


# ============================================================
# 1. 随机模拟每个候选分区点的延迟和能耗
# ============================================================
def simulate_latency_energy(n_points, seed=42):        #固定种子可复现，方便调试
    """
    模拟 n_points 个候选分区点的延迟 L 和能耗 E。
    真实场景中这些值来自在线采集器；这里先用随机数代替，
    但要保证量级合理：延迟在 0~10 秒，能耗在 0~3000 mWh。
    """
    rng = np.random.default_rng(seed)
    L = rng.uniform(0.5, 10.0, size=n_points)          # 延迟/秒
    E = rng.uniform(100.0, 3000.0, size=n_points)      # 能耗/mWh
    return L, E


# ============================================================
# 2. 标准化函数（论文式 7、8）
# ============================================================
def standardize_latency(L, w1=1.0, b1=-3.6, j1=1.3, k1=1.0):
    return -np.tanh((w1 * L + b1) / j1) + k1           #tanh 把任意数值压到 (-1, 1)


def standardize_energy(E, w2=1.0, b2=-2.1, j2=0.5, k2=1.0):
    return -np.tanh((w2 * E + b2) / j2) + k2


# ============================================================
# 3. 训练循环
# ============================================================
def train(num_episodes=1000, n_points=18, alpha=0.5, beta=0.5, lr=1e-4): #轮、候选点、权重、学习率
    """
    α 大（0.7~0.95）	   更看重延迟，宁可多耗电也要快	 实时控制、自动驾驶、交互应用
    α 小（0.05~0.3）	   延迟不那么重要	电池供电的传感器
    α = 0.5                延迟和能耗平衡   通用场景

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 模拟所有候选点的 L、E
    L_all, E_all = simulate_latency_energy(n_points)

    # 预先算好每个候选点的标准化效用值（作为"真实环境"的反馈）
    U_all = []
    for i in range(n_points):
        L_prime = standardize_latency(L_all[i])
        E_prime = standardize_energy(E_all[i])
        U_all.append(alpha * L_prime + beta * E_prime)
    U_all = np.array(U_all, dtype=np.float32)
    print("各候选点效用值范围:", U_all.min(), "~", U_all.max())

    # 初始化 DNN_P
    dnn_p = DNN_P(input_dim=3, output_dim=n_points).to(device)
    optimizer = torch.optim.Adam(dnn_p.parameters(), lr=lr)

    # 基线：历史效用值的平均
    baseline_list = []
    loss_history = []
    U_history = []

    for episode in range(num_episodes):
        # 1) 随机生成一个状态 s = [R, Q, h]       #实时上传传输速率transmission rate(大就说明网络快，传中间张量便宜，可以多切，让服务器多跑)
                                                 #链路信号质量signal quality、信道状态channel state
        R = np.random.uniform(1.0, 100.0)
        Q = np.random.uniform(0.0, 1.0)
        h = np.random.uniform(-90.0, -30.0)     #越接近 0，信号越好
        s = torch.tensor([[R / 100.0, Q, (h + 100.0) / 100.0]],#双层方括号是二维，手动归一化，防止网格难收敛
                         dtype=torch.float32, device=device)   #tensor创建一个shape = (1, 3)的pytorch张量

        # 2) DNN-p 前向，得到概率分布
        o, p = dnn_p(s)                       # p: (1, n_points)
        p_np = p.detach().cpu().numpy().flatten()
                #把张量从计算图里摘出来，方便后续的计算；把张量移到CPU上，因为numpy数组只能存在CPU上；
                # 把 PyTorch 张量转成 NumPy 数组；多维数组压成一维数组，方便 np.random.choice() 采样
       
        # 3) 按概率采样分区点 a
        a = np.random.choice(n_points, p=p_np)

        # 4) 从模拟环境获取该分区点的效用值
        U = U_all[a]

        # 5) 计算损失（式 6）
        baseline = np.mean(baseline_list) if len(baseline_list) > 0 else 0.0  #历史所有任务效用值的平均值
        log_p_a = torch.log(p[0, a] + 1e-8)     #加一个极小的 1e-8，保证 log 的参数永远大于 0；概率越小，log越负
        loss = -log_p_a * (U - baseline)        #如果 U < baseline（比平均好），这一项为负，梯度方向会增大 p_a；
                                                #如果 U > baseline（比平均差），会减小 p_a。

        # 6) 反向传播 + 更新（清空梯度 → 反向传播 → 更新参数）
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 7) 更新基线
        baseline_list.append(U)
        loss_history.append(loss.item())
        U_history.append(U)

        if episode % 100 == 0:
            print(f"Episode {episode:4d} | a={a:2d} | U={U:.4f} | "
                  f"baseline={baseline:.4f} | loss={loss.item():.4f}")

    return loss_history, U_history


# ============================================================
# 4. 画图
# ============================================================
def plot_curves(loss_history, U_history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(loss_history)
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True, alpha=0.3)

    # 平滑后的效用值曲线
    window = 20
    U_smooth = np.convolve(U_history, np.ones(window) / window, mode="valid")
    axes[1].plot(U_smooth, color="orange")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Utility (smoothed)")
    axes[1].set_title("Utility over Training")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("train_sim_curves.png", dpi=150)
    plt.show()
    print("曲线已保存到 train_sim_curves.png")


if __name__ == "__main__":
    loss_history, U_history = train(num_episodes=1000, n_points=18)
    plot_curves(loss_history, U_history)