import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 输入数据（x为右边值，y为左边值）
x = np.array([2, 1.8, 1.6, 1.5,1.4, 1.2, 1])
y = np.array([26.4, 24.7, 22.7, 21.5,20.2, 15.9, 5.9])

# 定义指数函数模型：y = a * exp(b*x) + c
def exp_func(x, a, b, c):
    return c*np.log(a * x+b)

# 进行曲线拟合
p0 = [1, 1, 1]  # 初始参数猜测
params, covariance = curve_fit(exp_func, x, y, p0=p0, maxfev=100000)

# 提取拟合参数
a, b, c = params
print(f"拟合方程：y = {c:.3e} * log({a:.5f}*x) + {b:.3f}")

# 计算R平方值
residuals = y - exp_func(x, *params)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y - np.mean(y))**2)
r_squared = 1 - (ss_res / ss_tot)
print(f"R² = {r_squared:.4f}")

# 生成拟合曲线
x_fit = np.linspace(1, 2, 200)
y_fit = exp_func(x_fit, *params)

# 可视化结果
plt.figure(figsize=(10, 6))
plt.scatter(x, y, s=100, label='origin data', zorder=3, edgecolor='k')
plt.plot(x_fit, y_fit, 'r-', label=f'y = {c:.3f}ln{{{a:.5f}x{b:.3f}}}')
plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)
plt.title('(R² = {:.4f})'.format(r_squared))
plt.grid(alpha=0.3)
plt.legend()
plt.show()