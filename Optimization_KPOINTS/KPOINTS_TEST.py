import os
import numpy as np
import matplotlib.pyplot as plt
from ase.io.vasp import read_vasp_out

# 设置matplotlib中文字体（如果需要显示中文）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def extract_energy_from_outcar(outcar_path):
    """
    从OUTCAR文件中提取能量信息
    """
    try:
        # 使用ase库读取OUTCAR文件
        atoms = read_vasp_out(outcar_path, index=-1)
        energy = atoms.get_potential_energy()
        return energy
    except:
        # 如果ase读取失败，尝试手动解析
        try:
            with open(outcar_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if 'free  energy   TOTEN' in line:
                        energy = float(line.split()[-2])
                        return energy
        except:
            print(f"无法从 {outcar_path} 提取能量信息")
            return None


def get_kpoints_from_folder(folder_path):
    """
    从文件夹名提取KPOINTS值
    """
    try:
        # 假设文件夹名格式为 KPOINTS_X
        kpoints = int(folder_path.split('_')[-1])
        return kpoints
    except:
        print(f"无法从 {folder_path} 提取KPOINTS值")
        return None


def plot_kpoints_convergence():
    """
    绘制KPOINTS收敛测试图
    """
    # 存储结果的字典
    results = {}

    # 遍历所有KPOINTS文件夹
    base_dirs = ['KPOINTS_3', 'KPOINTS_5', 'KPOINTS_7',
                 'KPOINTS_9', 'KPOINTS_11', 'KPOINTS_13', 'KPOINTS_15',]

    for folder in base_dirs:
        if os.path.exists(folder):
            outcar_path = os.path.join(folder, 'OUTCAR')
            if os.path.exists(outcar_path):
                kpoints = get_kpoints_from_folder(folder)
                energy = extract_energy_from_outcar(outcar_path)

                if kpoints is not None and energy is not None:
                    results[kpoints] = energy
                    print(f"KPOINTS={kpoints}: 能量 = {energy:.6f} eV")

    if not results:
        print("未找到任何有效的OUTCAR文件")
        return

    # 按KPOINTS值排序
    sorted_kpoints = sorted(results.keys())
    sorted_energies = [results[k] for k in sorted_kpoints]

    # 计算能量相对于最大KPOINTS值的差值
    ref_energy = sorted_energies[-1]  # 使用最大KPOINTS值的能量作为参考
    energy_differences = [abs(e - ref_energy) for e in sorted_energies]

    # 创建图表
    fig, (ax1) = plt.subplots(1, figsize=(12, 5))

    # 绘制能量随KPOINTS变化图
    ax1.plot(sorted_kpoints, sorted_energies, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('KPOINTS')
    ax1.set_ylabel('Energy(eV)')
    ax1.set_title('KPOINTS')
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('kpoints_convergence.png', dpi=300)
    plt.show()

    # 输出收敛分析
    print("\n收敛分析:")
    for i, k in enumerate(sorted_kpoints):
        print(f"KPOINTS={k}: 与参考能量差 = {energy_differences[i]:.6f} eV")

    # 寻找收敛点（通常定义为能量变化小于某个阈值，例如0.001 eV）
    threshold = 0.001  # 1 meV的阈值
    converged_kpoints = None
    for i, k in enumerate(sorted_kpoints):
        if energy_differences[i] < threshold:
            converged_kpoints = k
            print(f"\n收敛点: KPOINTS = {k} (能量差 < {threshold} eV)")
            break

    if converged_kpoints is None:
        print(f"\n在测试范围内未找到收敛点（阈值 = {threshold} eV）")


if __name__ == "__main__":
    plot_kpoints_convergence()