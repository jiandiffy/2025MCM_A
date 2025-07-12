import numpy as np
import matplotlib.pyplot as plt

# 1. 时间范围（0–100 年）
t = np.linspace(0, 100, 101)          # 共 101 个点，便于画平滑曲线

# 2. 各国参数（日均人流 Nd、风化速率常数 p、曲线颜色）
params = {
    "China":    {"Nd": 1000, "p": 0.005, "color": "tab:blue"},
    "Brazil":   {"Nd":  800, "p": 0.012, "color": "tab:orange"},
    "Scotland": {"Nd":  260, "p": 0.008, "color": "tab:green"},
    "America":  {"Nd":  500, "p": 0.005, "color": "tab:purple"},
}

# 3. 缩放系数（经验值，使曲线数量级与原图一致）
k = 1e-9   # (m / year²) per (person · p)

# 4. 绘图
plt.figure(figsize=(6, 4))            # 6×4 英寸画布

for country, v in params.items():
    # 累计磨损体积：wear(t) = k * Nd * p * t²
    wear = k * v["Nd"] * v["p"] * t**2
    plt.plot(t, wear, label=country, color=v["color"])
    # 半透明填充，增强视觉效果
    plt.fill_between(t, 0, wear, alpha=0.15, color=v["color"])

# 5. 图形修饰
plt.xlabel("Time (years)")
plt.ylabel("Cumulative Wear volume (m)")
plt.title("Wear accumulation by country")
plt.legend(title="Country Parameters")
plt.xlim(0, 100)
plt.ylim(0)                           # y 轴从 0 开始
plt.grid(alpha=0.3)
plt.tight_layout()

# 6. 显示
plt.show()
