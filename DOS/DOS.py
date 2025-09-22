import numpy as np
import matplotlib.pyplot as plt
import re
import os


def extract_fermi_energy(filename):
    """从文件中提取费米能级值，处理各种可能的格式"""
    if not os.path.exists(filename):
        print(f"警告：文件 {filename} 不存在")
        return 0.0

    with open(filename, 'r') as f:
        content = f.read().strip()

    # 尝试直接转换为浮点数
    try:
        return float(content)
    except ValueError:
        pass

    # 尝试从文本中提取数值
    matches = re.findall(r'[-+]?\d*\.\d+|\d+', content)
    if matches:
        try:
            return float(matches[0])
        except ValueError:
            pass

    # 尝试从最后一行提取数值
    lines = content.splitlines()
    if lines:
        try:
            return float(lines[-1])
        except ValueError:
            pass

    # 如果所有方法都失败，尝试使用默认值
    print(f"警告：无法从 {filename} 文件中提取费米能级值")
    print("文件内容：")
    print(content)
    print("-" * 50)
    print("使用默认值 0.0 eV")
    return 0.0


# 读取费米能级
fermi_energy = extract_fermi_energy('FERMI_ENERGY')
print(f"使用的费米能级: {fermi_energy} eV")

# 读取总态密度 (TDOS.dat)
if os.path.exists('TDOS.dat'):
    with open('TDOS.dat', 'r') as f:
        lines = []
        for line in f:
            # 跳过注释行和空行
            if line.strip() and not line.startswith('#'):
                lines.append(line)

        # 检查是否有标题行
        if lines and any(char.isalpha() for char in lines[0]):
            header = lines[0]
            data_lines = lines[1:]
        else:
            data_lines = lines

        tdos_data = np.loadtxt(data_lines)

    # energy = tdos_data[:, 0] - fermi_energy  # 将费米能级设为0点
    energy = tdos_data[:, 0]  # 将费米能级设为0点
    total_dos = tdos_data[:, 1]  # 总态密度
else:
    print("警告：TDOS.dat 文件不存在")
    energy = np.array([])
    total_dos = np.array([])

# 读取投影态密度 (PDOS_USER.dat)
if os.path.exists('PDOS_USER.dat'):
    with open('PDOS_USER.dat', 'r') as f:
        lines = []
        for line in f:
            # 跳过注释行和空行
            if line.strip() and not line.startswith('#'):
                lines.append(line)

        # 检查是否有标题行
        if lines and any(char.isalpha() for char in lines[0]):
            header = lines[0]
            data_lines = lines[1:]
        else:
            data_lines = lines

        pdos_data = np.loadtxt(data_lines)

    # pdos_energy = pdos_data[:, 0] - fermi_energy
    pdos_energy = pdos_data[:, 0]

    # 提取各轨道PDOS
    try:
        # 尝试不同列索引方案
        if pdos_data.shape[1] >= 8:
            # 格式：能量 | s↑ | s↓ | py↑ | py↓ | pz↑ | pz↓ | px↑ | px↓ | ...
            s_dos = pdos_data[:, 1]  # s轨道
            px_dos = pdos_data[:, 3]  # py轨道
            py_dos = pdos_data[:, 5]  # pz轨道
            pz_dos = pdos_data[:, 7]  # px轨道
        elif pdos_data.shape[1] >= 5:
            # 格式：能量 | s | py | pz | px
            s_dos = pdos_data[:, 1]  # s轨道
            px_dos = pdos_data[:, 2]  # py轨道
            py_dos = pdos_data[:, 3]  # pz轨道
            pz_dos = pdos_data[:, 4]  # px轨道
        else:
            print("警告：PDOS文件列数不足")
            s_dos = np.zeros_like(pdos_energy)
            py_dos = np.zeros_like(pdos_energy)
            pz_dos = np.zeros_like(pdos_energy)
            px_dos = np.zeros_like(pdos_energy)
    except IndexError:
        print("警告：无法解析PDOS文件格式")
        s_dos = np.zeros_like(pdos_energy)
        py_dos = np.zeros_like(pdos_energy)
        pz_dos = np.zeros_like(pdos_energy)
        px_dos = np.zeros_like(pdos_energy)
else:
    print("警告：PDOS_USER.dat 文件不存在")
    pdos_energy = np.array([])
    s_dos = np.array([])
    py_dos = np.array([])
    pz_dos = np.array([])
    px_dos = np.array([])

# =====================================================
# 创建并排的子图
# =====================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 确定能量范围
all_energy = []
if len(energy) > 0:
    all_energy.extend(energy)
if len(pdos_energy) > 0:
    all_energy.extend(pdos_energy)

if all_energy:
    y_min = min(all_energy)
    y_max = max(all_energy)
else:
    y_min, y_max = -10, 10  # 默认范围

# =====================================================
# 左图：总态密度图 (Total DOS)
# =====================================================
if len(energy) > 0:
    ax1.plot(total_dos, energy, 'k-', linewidth=2.0, label='Total DOS')
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=1.5, alpha=0.8)
    ax1.set_xlabel('DOS (states/eV)', fontsize=12)
    ax1.set_ylabel('Energy (eV)', fontsize=12)
    ax1.set_title('Total Density of States', fontsize=16)
    ax1.set_xlim(0, max(total_dos) * 1.1 if max(total_dos) > 0 else 1)
    ax1.set_ylim(y_min, y_max)
    ax1.legend(frameon=False, fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.6)
else:
    ax1.text(0.5, 0.5, "无有效数据",
             horizontalalignment='center',
             verticalalignment='center',
             transform=ax1.transAxes,
             fontsize=14)
    ax1.set_title('Total Density of States (无数据)', fontsize=16)
    ax1.set_ylim(y_min, y_max)

# =====================================================
# 右图：分轨道态密度图 (Projected DOS)
# =====================================================
if len(pdos_energy) > 0:
    # 计算最大PDOS值用于设置x轴范围
    max_pdos = max(np.max(s_dos), np.max(px_dos), np.max(py_dos), np.max(pz_dos))

    ax2.plot(s_dos, pdos_energy, 'b-', linewidth=1.8, label='s')
    ax2.plot(px_dos, pdos_energy, 'g-', linewidth=1.8, label='px')
    ax2.plot(py_dos, pdos_energy, 'y-', linewidth=1.8, label='py')
    ax2.plot(pz_dos, pdos_energy, 'r-', linewidth=1.8, label='pz')
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=1.5, alpha=0.8)
    ax2.set_xlabel('PDOS (states/eV)', fontsize=12)
    ax2.set_title('Projected Density of States', fontsize=16)
    ax2.set_xlim(0, max_pdos * 1.2 if max_pdos > 0 else 1)
    ax2.set_ylim(y_min, y_max)
    ax2.legend(frameon=False, fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.6)
    # 隐藏右图的y轴标签，因为左图已经有了
    ax2.set_ylabel('')
else:
    ax2.text(0.5, 0.5, "无有效数据",
             horizontalalignment='center',
             verticalalignment='center',
             transform=ax2.transAxes,
             fontsize=14)
    ax2.set_title('Projected Density of States (无数据)', fontsize=16)
    ax2.set_ylim(y_min, y_max)

plt.tight_layout()
plt.savefig('Combined_DOS.png', dpi=300, bbox_inches='tight')
print("已保存合并的态密度图: Combined_DOS.png")
plt.close()

print("绘图完成！")