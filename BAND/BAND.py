import numpy as np
import matplotlib.pyplot as plt

# 加载能带数据
band_data = np.loadtxt('REFORMATTED_BAND.dat')
k_points = band_data[:, 0]  # K点路径
energies = band_data[:, 1:]  # 能量值

# 加载高对称点
sym_points = []
labels = []

with open('KLABELS', 'r') as f:
    lines = f.readlines()
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                coord = float(parts[1])
                sym_points.append(coord)
                label = parts[0]
                if label == 'Gamma':
                    label = r'\Gamma'
                labels.append(label)
            except (ValueError, IndexError):
                continue

print("高对称点坐标:", sym_points)
print("高对称点标签:", labels)

# 创建绘图
plt.figure(figsize=(12, 8), dpi=100)
plt.style.use('seaborn-v0_8-whitegrid')  # 使用白色网格背景

# 绘制所有能带（散点图）- 使用更深更鲜艳的蓝色
for band in range(energies.shape[1]):
    plt.scatter(k_points, energies[:, band],
                color='blue',  # 更深更饱和的蓝色
                s=20,  # 进一步增大点的大小
                alpha=0.9,  # 减少透明度使颜色更明显
                edgecolors='none',
                marker='o')

# 添加高对称点标记线
for point in sym_points:
    plt.axvline(x=point, color='#ff6b6b', linestyle='--', alpha=0.9, linewidth=1.5)

# 设置费米能级参考线
plt.axhline(y=0, color='#ff0000', linestyle='-', alpha=1.0, linewidth=2.0)

# 设置坐标轴范围
plt.xlim(min(k_points), max(k_points))
plt.ylim(-20, 20)

# 设置高对称点标签
plt.xticks(sym_points, labels=[r'$' + label + '$' for label in labels], fontsize=14)

# 添加坐标轴标签
plt.xlabel(r'Wave Vector', fontsize=16, labelpad=10)
plt.ylabel(r'Energy (eV)', fontsize=16, labelpad=10)

# 添加标题
plt.title('Graphene Band Structure', fontsize=16, pad=10)

# 添加网格
plt.grid(True, linestyle='--', alpha=0.3)

# 优化布局
plt.tight_layout()
plt.subplots_adjust(bottom=0.12, top=0.92)

# 保存和显示
plt.savefig('graphene_band_scatter_vibrant.png', dpi=300, bbox_inches='tight')
plt.show()