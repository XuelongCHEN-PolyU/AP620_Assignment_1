import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import re

# 设置全局字体和绘图参数
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['mathtext.fontset'] = 'dejavusans'


def parse_band_data(filename):
    """解析能带数据文件"""
    bands = {}
    current_band = None
    data_lines = []

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 检测新能带开始
            if line.startswith("# Band-Index"):
                # 保存上一个能带的数据
                if current_band is not None and data_lines:
                    band_array = np.array([list(map(float, line.split())) for line in data_lines])
                    bands[current_band] = band_array
                    data_lines = []

                # 解析当前能带编号
                match = re.search(r"# Band-Index\s+(\d+)", line)
                if match:
                    current_band = int(match.group(1))
                continue

            # 跳过标题行
            if line.startswith("#K-Path") or not line[0].isdigit():
                continue

            data_lines.append(line)

        # 处理最后一个能带
        if current_band is not None and data_lines:
            band_array = np.array([list(map(float, line.split())) for line in data_lines])
            bands[current_band] = band_array

    return bands


# 解析数据文件 (替换为您的实际文件名)
filename = "PBAND_C.dat"
bands_data = parse_band_data(filename)

# 创建图形
plt.figure(figsize=(10, 8), dpi=300)

# 轨道颜色映射
colors = {
    's': '#FF0000',  # 红色 - s轨道
    'px': '#0000FF',  # 蓝色 - px轨道
    'py': '#00CC00',  # 绿色 - py轨道
    'pz': '#800080'  # 紫色 - pz轨道
}

# 轨道标记大小缩放因子
marker_scale = 50

# 绘制每个能带
for band_idx, band_data in bands_data.items():
    # 提取数据
    kpoints = band_data[:, 0]  # 第一列：k点路径
    energies = band_data[:, 1]  # 第二列：能量值
    s_data = band_data[:, 2]  # 第三列：s轨道投影
    py_data = band_data[:, 3]  # 第四列：py轨道投影
    pz_data = band_data[:, 4]  # 第五列：pz轨道投影
    px_data = band_data[:, 5]  # 第六列：px轨道投影

    # 绘制轨道投影
    plt.scatter(kpoints, energies, s=s_data * marker_scale, c=colors['s'],
                alpha=0.7, label='s' if band_idx == 1 else "", edgecolors='none')
    plt.scatter(kpoints, energies, s=px_data * marker_scale, c=colors['px'],
                alpha=0.7, label='px' if band_idx == 1 else "", edgecolors='none')
    plt.scatter(kpoints, energies, s=py_data * marker_scale, c=colors['py'],
                alpha=0.7, label='py' if band_idx == 1 else "", edgecolors='none')
    plt.scatter(kpoints, energies, s=pz_data * marker_scale, c=colors['pz'],
                alpha=0.7, label='pz' if band_idx == 1 else "", edgecolors='none')

    # 绘制能带线
    plt.plot(kpoints, energies, 'k-', lw=0.5, alpha=0.3)

# 设置图例
legend = plt.legend(loc='upper right', fontsize=10, frameon=True)
legend.get_frame().set_edgecolor('k')
legend.get_frame().set_linewidth(1.0)

# 设置坐标轴标签
plt.xlabel('k-Path', fontsize=14, fontweight='bold')
plt.ylabel('Energy (eV)', fontsize=14, fontweight='bold')

# 设置网格
plt.grid(True, linestyle='--', alpha=0.6)

# 设置坐标轴范围
plt.xlim(min(kpoints), max(kpoints))
plt.ylim(-20, 10)  # 根据数据调整Y轴范围

# 添加标题
plt.title('Orbital-projected Band Structure', fontsize=16, pad=15)

# 紧凑布局
plt.tight_layout()

# 保存图像
plt.savefig('band_structure.png', bbox_inches='tight')

# 显示图像
plt.show()