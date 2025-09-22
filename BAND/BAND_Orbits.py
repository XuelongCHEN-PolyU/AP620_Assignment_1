import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import os


# 文件读取函数
def read_klabels(filename):
    """
    读取KLABELS文件，获取高对称点信息和位置
    返回: 高对称点位置列表, 高对称点名称列表
    """
    kpoints = []
    labels = []

    with open(filename, 'r') as f:
        lines = f.readlines()

        # 跳过标题行
        for line in lines[1:]:
            if line.strip() and not line.startswith('*'):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        # 尝试将第二个部分转换为浮点数
                        kpoint = float(parts[1])
                        kpoints.append(kpoint)
                        labels.append(parts[0])
                    except ValueError:
                        # 如果转换失败，跳过这一行
                        continue

    return kpoints, labels


def read_pband(filename, num_orbitals=4):
    """
    读取PBAND_C.dat文件
    返回: k点坐标数组, 能量数组, 轨道投影数组
    """
    data = np.loadtxt(filename)
    k_path = data[:, 0]  # 第一列: k路径坐标
    energy = data[:, 1]  # 第二列: 能量
    # 剩余列: 轨道投影 (假设顺序为: s, py, pz, px)
    orbitals = data[:, 2:2 + num_orbitals]

    return k_path, energy, orbitals


# 主分析函数
def analyze_dirac_cone_contributions(pband_file, klabels_file, fermi_energy=0.0):
    """
    分析不同轨道对狄拉克锥的贡献
    """
    # 读取数据
    k_path, energy, orbitals = read_pband(pband_file)
    high_sym_k, high_sym_labels = read_klabels(klabels_file)

    # 找到K点位置
    k_point_idx = None
    for i, label in enumerate(high_sym_labels):
        if label.upper() == 'K':
            k_point = high_sym_k[i]
            # 找到最接近K点的索引
            k_point_idx = np.argmin(np.abs(k_path - k_point))
            break

    if k_point_idx is None:
        raise ValueError("未找到K点，请检查KLABELS文件")

    # 获取K点附近的数据
    k_range = 2  # K点附近的范围
    k_mask = (k_path > k_point - k_range) & (k_path < k_point + k_range)
    k_near = k_path[k_mask]
    energy_near = energy[k_mask]
    orbitals_near = orbitals[k_mask]

    # 找到费米能级附近的点
    e_range = 2  # 费米能级附近的能量范围
    e_mask = (energy_near > fermi_energy - e_range) & (energy_near < fermi_energy + e_range)

    if np.sum(e_mask) == 0:
        print("警告: 在K点附近未找到费米能级附近的点，尝试扩大能量范围")
        e_range = 0.5
        e_mask = (energy_near > fermi_energy - e_range) & (energy_near < fermi_energy + e_range)

    # 计算各轨道在狄拉克点附近的平均贡献
    orbital_contributions = np.mean(orbitals_near[e_mask], axis=0)
    total_contribution = np.sum(orbital_contributions)

    # 计算相对贡献百分比
    orbital_percentages = 100 * orbital_contributions / total_contribution

    return orbital_percentages, k_near, energy_near, orbitals_near


# 可视化函数
# 可视化函数 - 使用柱状图展示轨道贡献
def plot_dirac_cone_and_contributions(k_near, energy_near, orbitals_near, orbital_percentages):
    """
    绘制狄拉克锥和轨道贡献柱状图
    """
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 左图：绘制狄拉克锥
    scatter = ax1.scatter(k_near, energy_near, c=orbitals_near[:, 2], cmap='viridis', s=10)
    ax1.set_xlabel('k-path')
    ax1.set_ylabel('Energy (eV)')
    ax1.set_title('Dirac Cone at K point')
    plt.colorbar(scatter, ax=ax1, label='pz orbital weight')

    # 右图：绘制轨道贡献柱状图
    labels = ['s', 'py', 'pz', 'px']
    colors = ['red', 'blue', 'green', 'purple']

    # 创建柱状图
    bars = ax2.bar(labels, orbital_percentages, color=colors)
    ax2.set_xlabel('Orbitals')
    ax2.set_ylabel('Contribution Percentage (%)')
    ax2.set_title('Orbital Contributions to Dirac Cone')

    # 在柱子上方添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                 f'{height:.1f}%', ha='center', va='bottom')

    # 可选：添加网格线以便更清晰地读取数值
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    # 设置y轴范围，使标签显示更舒适
    ax2.set_ylim(0, max(orbital_percentages) * 1.1)

    plt.tight_layout()
    plt.savefig('dirac_cone_analysis.png', dpi=300)
    plt.show()


# 绘制完整能带结构函数 - 显示所有轨道
def plot_full_band_structure(pband_file, klabels_file, fermi_energy=0.0):
    """
    绘制完整的能带结构图，显示所有轨道的贡献
    """
    # 读取数据
    k_path, energy, orbitals = read_pband(pband_file)
    high_sym_k, high_sym_labels = read_klabels(klabels_file)

    # 创建图形和子图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # 轨道名称和颜色
    orbital_names = ['s', 'py', 'pz', 'px']
    colors = ['red', 'blue', 'green', 'purple']

    # 绘制每个轨道的能带结构
    for i, (ax, name, color) in enumerate(zip(axes, orbital_names, colors)):
        # 绘制能带结构，颜色表示当前轨道的权重
        scatter = ax.scatter(k_path, energy - fermi_energy, c=orbitals[:, i],
                             cmap='viridis', s=5, alpha=0.7)

        # 添加高对称点标记
        for k, label in zip(high_sym_k, high_sym_labels):
            ax.axvline(x=k, color='gray', linestyle='--', alpha=0.5)
            if i >= 2:  # 只在底部两个子图添加x轴标签
                ax.text(k, ax.get_ylim()[0] - 0.5, label, ha='center')

        ax.set_title(f'{name} orbital contribution')
        ax.set_ylabel('Energy - E_F (eV)')

        # 添加颜色条
        plt.colorbar(scatter, ax=ax, label=f'{name} orbital weight')

    # 设置x轴范围并添加x轴标签（只在底部子图）
    for i in range(2, 4):
        axes[i].set_xlabel('k-path')

    # 移除顶部子图的x轴刻度标签
    for i in range(0, 2):
        axes[i].tick_params(labelbottom=False)

    plt.suptitle('Graphene Band Structure - Orbital Contributions')
    plt.tight_layout()
    plt.savefig('full_band_structure_all_orbitals.png', dpi=300)
    plt.show()


# 主函数
def main():
    # 文件路径
    pband_file = 'PBAND_C.dat'
    klabels_file = 'KLABELS'

    # 检查文件是否存在
    if not os.path.exists(pband_file):
        print(f"错误: 找不到文件 {pband_file}")
        return
    if not os.path.exists(klabels_file):
        print(f"错误: 找不到文件 {klabels_file}")
        return

    # 分析狄拉克锥贡献
    try:
        orbital_percentages, k_near, energy_near, orbitals_near = analyze_dirac_cone_contributions(
            pband_file, klabels_file)

        # 打印结果
        orbital_names = ['s', 'py', 'pz', 'px']
        print("各轨道对狄拉克锥的贡献百分比:")
        for name, percentage in zip(orbital_names, orbital_percentages):
            print(f"{name}轨道: {percentage:.2f}%")

        # 绘制图形
        plot_dirac_cone_and_contributions(k_near, energy_near, orbitals_near, orbital_percentages)

        # 绘制完整能带结构
        plot_full_band_structure(pband_file, klabels_file)

        # 确定主要贡献轨道
        main_orbital_idx = np.argmax(orbital_percentages)
        print(f"\n主要贡献轨道: {orbital_names[main_orbital_idx]}轨道")

    except Exception as e:
        print(f"分析过程中出现错误: {e}")


if __name__ == "__main__":
    main()