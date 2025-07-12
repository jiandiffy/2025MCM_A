import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
# 1) 定义自定义Colormap: 从天蓝(#87CEEB)到粉色(#FFC0CB)
cmap_blue_pink = LinearSegmentedColormap.from_list(
    'blue_pink',  # colormap 名字，可自定
    ['#87CEEB', '#FFC0CB']
)
T=100
N_d=260
G=700
K=1.8e-7
d=8e-4
H=175e6
k_m=N_d*G*K*d/H
# 1) 定义自定义Colormap: 从天蓝(#87CEEB)到粉色(#FFC0CB)
cmap_blue_pink = LinearSegmentedColormap.from_list(
    'blue_pink',  # colormap 名字，可自定
    ['#87CEEB', '#FFC0CB']
)
############################################
# 1.  假设我们有如下函数来给出：
#     - 2D 热力图数据 f2d
#     - 一维分布 X(x) 与 Y(y)
############################################
def X(x):
    """示例: 人为定义的 3个高斯混合, 仅供演示."""
    from math import exp, sqrt, pi
    # (此处写死, 真实场景可替换成您自己的 X(x))
    w = [0.3, 0.4, 0.3]
    mu= [3.0, 3.0, 3.0]
    s = [2, 2, 2]
    total = 0
    for i in range(3):
        total += w[i]*(1/(sqrt(2*pi)*s[i]))*exp(-0.5*((x-mu[i])/s[i])**2)
    return total

def Y(y):
    """示例: 人为定义的 2个高斯混合, 仅供演示."""
    from math import exp, sqrt, pi
    w = [0.4, 0.6]
    mu= [1.0, 1.0]
    s = [2.5, 2.5]
    total = 0
    for j in range(2):
        total += w[j]*(1/(sqrt(2*pi)*s[j]))*exp(-0.5*((y-mu[j])/s[j])**2)
    return total



T_list = [1e6, 2e6, 3e6, 4e6, 5e6, 7e6]

########################################
# 2. 生成网格
########################################
Nx, Ny = 80, 80
xs = np.linspace(0, 5, Nx)
ys = np.linspace(0, 5, Ny)

########################################
# 3. 依次计算每个 T 下的 f2d(x,y)
#    并收集到 all_data 里
########################################
all_data = []
for T in T_list:
    f2d = np.zeros((Nx, Ny))
    for i, xv in enumerate(xs):
        for j, yv in enumerate(ys):
            f2d[i, j] = X(xv) * Y(yv) * k_m * T
    all_data.append((T, f2d))

########################################
# 4. 找到所有 f2d 的 全局 min/max
########################################
global_min = min( np.min(d) for (_, d) in all_data )
global_max = max( np.max(d) for (_, d) in all_data )

print("Global min:", global_min, " Global max:", global_max)

########################################
# 5. 分别画出 6 张图 (每个 T 一个Figure)
########################################
for T, f2d in all_data:
    plt.figure(figsize=(6,5))  # 建立新的 Figure
    # 绘制 heatmap, 并指定相同的vmin/vmax
    im = plt.imshow(
        f2d.T,
        origin='lower',
        extent=[xs[0], xs[-1], ys[0], ys[-1]],
        cmap=cmap_blue_pink , # 使用自定义的蓝→粉渐变
        vmin=global_min,
        vmax=global_max
    )
    plt.colorbar(im, label="f(x,y)")
    plt.title(f"Heatmap for T = {T:.0e}")
    plt.xlabel("x")
    plt.ylabel("y")
    ax=plt.gca
    ax.set_aspect(0.2, adjustable='box')
    plt.show()  # 显示这一张图；若您只想一次性结束后再看，也可不写