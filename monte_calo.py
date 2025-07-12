# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

# 设置随机数种子（可选，用于结果复现）
np.random.seed(42)

# 1. 参数设定

# 台阶尺寸（单位：厘米）
step_length = 100.0  # 长度方向：1米
step_width = 20.0    # 宽度方向：20厘米

# 脚的尺寸（单位：厘米）
foot_length = 15.0    # 脚的长度：15厘米
foot_width = 5.0      # 脚的宽度：5厘米

# 脚心最小距离约束（单位：厘米）
min_center_distance = 40.0

# 正态分布参数
mu_x = 50.0       # 长度方向均值（中心位置）
sigma_x = 10.0    # 长度方向标准差
mu_y = 10.0       # 宽度方向均值（中心位置）
sigma_y = 3.0     # 宽度方向标准差

# 人数与时间步
N_time_steps = 1000       # 时间步总数
n_arrivals_per_step = 5   # 每个时间步到达的人数（可根据需要修改）

# 脚心活动范围（确保脚完全在台阶上）
x_min = foot_length / 2
x_max = step_length - foot_length / 2
y_min = foot_width / 2
y_max = step_width - foot_width / 2

# 2. 定义函数

def get_truncated_normal(mean, sd, low, high):
    """
    获取截断正态分布的随机数生成器
    """
    return truncnorm(
        (low - mean) / sd,
        (high - mean) / sd,
        loc=mean,
        scale=sd)

# 创建截断正态分布的采样器
trunc_norm_x = get_truncated_normal(mu_x, sigma_x, x_min, x_max)
trunc_norm_y = get_truncated_normal(mu_y, sigma_y, y_min, y_max)

def sample_one_foot_center(foot_centers, min_dist):
    """
    采样一个满足距离约束的脚心位置
    """
    while True:
        x = trunc_norm_x.rvs()
        y = trunc_norm_y.rvs()
        # 检查与已有脚心的距离
        if all(np.sqrt((x - fx)**2 + (y - fy)**2) >= min_dist for fx, fy in foot_centers):
            return (x, y)

# 3. 蒙特卡洛模拟

# 存储所有脚的位置
all_foot_positions = []

for t in range(N_time_steps):
    foot_centers_this_step = []
    for i in range(n_arrivals_per_step):
        # 尝试采样一个脚心位置
        try:
            x, y = sample_one_foot_center(foot_centers_this_step, min_center_distance)
            foot_centers_this_step.append((x, y))
        except:
            # 如果无法采样到合适的位置，可以选择跳过或记录失败
            print(f"时间步 {t}，第 {i} 人无法放置脚。")
            continue
    # 将这一时间步的脚位置加入总列表
    all_foot_positions.extend(foot_centers_this_step)

# 转换为 NumPy 数组，便于处理
all_foot_positions = np.array(all_foot_positions)
xs = all_foot_positions[:, 0]
ys = all_foot_positions[:, 1]

# 4. 结果可视化

# 绘制脚心位置的二维直方图（热图）
plt.figure(figsize=(10, 4))
heatmap, xedges, yedges, img = plt.hist2d(
    xs, ys,
    bins=[50, 10],  # 可以根据需要调整
    range=[[0, step_length], [0, step_width]],
    cmap='viridis'
)
plt.colorbar(label='踩脚次数')
plt.xlabel('长度方向位置（厘米）')
plt.ylabel('宽度方向位置（厘米）')
plt.title('脚心在台阶上的分布热图')
plt.tight_layout()
plt.show()

# 可选：显示脚心分布的散点图
plt.figure(figsize=(10, 4))
plt.scatter(xs, ys, alpha=0.1, s=10)
plt.xlabel('长度方向位置（厘米）')
plt.ylabel('宽度方向位置（厘米）')
plt.title('脚心在台阶上的分布散点图')
plt.xlim(0, step_length)
plt.ylim(0, step_width)
plt.tight_layout()
plt.show()

# 5. 频率统计（可选）

# 使用二维直方图统计频率
counts, xedges, yedges = np.histogram2d(
    xs, ys,
    bins=[50, 10],
    range=[[0, step_length], [0, step_width]]
)

# 打印某个区域的频率，例如中心区域
center_x = step_length / 2
center_y = step_width / 2
# 找到最近的bin
x_bin = np.digitize(center_x, xedges) - 1
y_bin = np.digitize(center_y, yedges) - 1
print(f"中心区域踩脚次数: {counts[x_bin, y_bin]}")

# 6. 保存结果（可选）

# 将频率数据保存为图像或数据文件
# 例如，保存热图
plt.figure(figsize=(10, 4))
plt.imshow(
    counts.T,
    origin='lower',
    extent=[0, step_length, 0, step_width],
    aspect='auto',
    cmap='viridis'
)
plt.colorbar(label='踩脚次数')
plt.xlabel('长度方向位置（厘米）')
plt.ylabel('宽度方向位置（厘米）')
plt.title('脚心在台阶上的分布热图')
plt.tight_layout()
plt.savefig('foot_position_heatmap.png', dpi=300)
plt.show()


