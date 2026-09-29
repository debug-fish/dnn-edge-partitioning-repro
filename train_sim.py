import torch
import numpy as np
import matplotlib.pyplot as plt

from dnn_p import DNN_P


# ============================================================
# 1. 随机模拟每个候选分区点的延迟和能耗
# ============================================================
def simulate_latency_energy(n_points, seed=42):
    """
    模拟 n_points 个候选分区点的延迟 L 和能耗 E。
    真实场景中这些值来自在线采集器；这里先用随机数代替，
    但要保证量级合理：延迟在 0~10 秒，能耗在 0~3000 mWh。
    """
    rng = np.random.default_rng(seed)
    L = rng.uniform(0.5, 10.0, size=n_points)          # 秒
    E = rng.uniform(100.0, 3000.0, size=n_points)      # mWh
    return L, E


# ============================================================
# 2. 标准化函数（论文式 7、8）
# ============================================================
def standardize_latency(L, w1=1.0, b1=-3.6, j1=1.3, k1=1.0):
    return -np.tanh((w1 * L + b1) / j1) + k1


def standardize_energy(E, w2=1.0, b2=-2.1, j2=0.5, k2=1.0):
    return -np.tanh((w2 * E + b2) / j2) + k2


# ============================================================
# 3. 训练循环
# ============================================================
def train(num_episodes=1000, n_points=18, alpha=0.5, beta=0.5, lr=1e-4):
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
        # 1) 随机生成一个状态 s = [R, Q, h]
        R = np.random.uniform(1.0, 100.0)
        Q = np.random.uniform(0.0, 1.0)
        h = np.random.uniform(-90.0, -30.0)
        s = torch.tensor([[R / 100.0, Q, (h + 100.0) / 100.0]],
                         dtype=torch.float32, device=device)

        # 2) DNN-p 前向，得到概率分布
        o, p = dnn_p(s)                       # p: (1, n_points)
        p_np = p.detach().cpu().numpy().flatten()

        # 3) 按概率采样分区点 a
        a = np.random.choice(n_points, p=p_np)

        # 4) 从模拟环境获取该分区点的效用值
        U = U_all[a]

        # 5) 计算损失（式 6）
        baseline = np.mean(baseline_list) if len(baseline_list) > 0 else 0.0
        log_p_a = torch.log(p[0, a] + 1e-8)
        loss = -log_p_a * (U - baseline)

        # 6) 反向传播 + 更新
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