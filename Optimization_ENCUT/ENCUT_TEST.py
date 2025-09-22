import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from ase.io import read
from ase import Atoms

# 设置绘图样式
plt.style.use('default')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


def extract_energy_from_outcar(outcar_path):
    """从OUTCAR文件中提取能量"""
    energy = None
    with open(outcar_path, 'r') as f:
        for line in f:
            if 'free  energy   TOTEN' in line:
                parts = line.split()
                energy = float(parts[4])
                break
    return energy


def analyze_encut_convergence():
    """分析ENCUT收敛测试结果"""
    # ENCUT值列表
    encut_values = [200, 250, 300, 350, 400, 450 ,500, 550]
    energies = []

    # 从每个目录提取能量
    for encut in encut_values:
        outcar_path = f"Encut_{encut}/OUTCAR"
        if os.path.exists(outcar_path):
            energy = extract_energy_from_outcar(outcar_path)
            energies.append(energy)
            print(f"ENCUT = {encut} eV, Energy = {energy:.6f} eV")
        else:
            energies.append(np.nan)
            print(f"ENCUT = {encut} eV, OUTCAR not found")

    # 绘制ENCUT收敛曲线
    plt.figure(figsize=(10, 6))
    plt.plot(encut_values, energies, 'o-', linewidth=2, markersize=8, color='blue')
    plt.xlabel('ENCUT (eV)')
    plt.ylabel('Total Energy (eV)')
    plt.title('ENCUT Convergence Test for Graphene')
    plt.grid(True, alpha=0.3)

    # 标记收敛点
    if len(energies) > 1:
        # 计算能量差
        energy_diffs = np.abs(np.diff(energies))
        # 找到第一个能量差小于1 meV的点
        convergence_index = None
        for i, diff in enumerate(energy_diffs):
            if diff < 0.001:  # 1 meV收敛标准
                convergence_index = i + 1
                break

        if convergence_index is not None:
            plt.axvline(x=encut_values[convergence_index], color='red', linestyle='--',
                        label=f'Converged at {encut_values[convergence_index]} eV')
            plt.legend()
            print(f"ENCUT converged at {encut_values[convergence_index]} eV")
        else:
            print("ENCUT did not converge within tested range")

    plt.tight_layout()
    plt.savefig('encut_convergence.png', dpi=300)
    plt.show()

    return encut_values, energies


def explain_incar_parameters():
    """解释INCAR文件中的关键参数"""
    print("=" * 60)
    print("EXPLANATION OF KEY INCAR PARAMETERS")
    print("=" * 60)

    parameters = {
        "ENCUT": {
            "value": "520",
            "explanation": "平面波基组的截断能。更高的值意味着更精确的计算但计算成本更高。需要进行收敛测试以确定合适的值。"
        },
        "PREC": {
            "value": "Accurate",
            "explanation": "控制计算精度。'Accurate'提供最高精度，但计算成本更高。"
        },
        "ISMEAR": {
            "value": "0",
            "explanation": "确定k点积分的方法。0表示Gaussian smearing，适用于半导体和绝缘体。"
        },
        "SIGMA": {
            "value": "0.05",
            "explanation": "smearing的宽度（单位：eV）。较小的值提供更精确的结果，但可能导致收敛问题。"
        },
        "EDIFF": {
            "value": "1E-6",
            "explanation": "电子自洽循环的能量收敛标准。较小的值意味着更精确的结果。"
        },
        "EDIFFG": {
            "value": "-0.01",
            "explanation": "离子弛豫的力收敛标准。负值表示使用能量变化作为收敛标准。"
        },
        "IBRION": {
            "value": "2",
            "explanation": "离子弛豫算法。2表示使用共轭梯度法。"
        },
        "ISIF": {
            "value": "3",
            "explanation": "控制哪些晶胞参数被优化。3表示优化晶胞形状和体积。"
        },
        "NSW": {
            "value": "100",
            "explanation": "最大离子步数。如果收敛在此之前达到，计算将提前停止。"
        },
        "LREAL": {
            "value": ".FALSE.",
            "explanation": "控制实空间投影算子的使用。对于小体系，应设置为.FALSE.以获得更精确的结果。"
        },
        "LWAVE": {
            "value": ".FALSE.",
            "explanation": "控制是否写入WAVECAR文件。对于单点计算，可以设置为.FALSE.以节省磁盘空间。"
        },
        "LCHARG": {
            "value": ".FALSE.",
            "explanation": "控制是否写入CHGCAR文件。对于单点计算，可以设置为.FALSE.以节省磁盘空间。"
        },
        "LORBIT": {
            "value": "11",
            "explanation": "控制投影态密度(PDOS)的输出。11表示输出详细的原子和轨道投影信息。"
        }
    }

    for param, info in parameters.items():
        print(f"{param:10} = {info['value']:10} : {info['explanation']}")

    print("=" * 60)


def main():
    """主函数"""
    print("Analyzing VASP calculation results...")
    print()

    # 分析ENCUT收敛测试
    print("1. ENCUT CONVERGENCE TEST")
    print("-" * 40)
    encut_values, encut_energies = analyze_encut_convergence()
    print()

    # 解释INCAR参数
    print("3. INCAR PARAMETER EXPLANATION")
    print("-" * 40)
    explain_incar_parameters()
    print()

if __name__ == "__main__":
    main()