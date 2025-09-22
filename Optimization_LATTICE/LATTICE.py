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


def extract_lattice_constant_from_contcar(contcar_path):
    """从CONTCAR文件中提取晶格常数"""
    with open(contcar_path, 'r') as f:
        lines = f.readlines()
        # 读取晶格缩放因子
        scale_factor = float(lines[1].strip())
        # 读取晶格矢量
        a_vector = np.array([float(x) for x in lines[2].split()])
        # 计算晶格常数
        lattice_constant = np.linalg.norm(a_vector) * scale_factor
    return lattice_constant


def analyze_lattice_constant_optimization():
    """分析晶格常数优化结果"""
    # 假设您有不同缩放系数的目录结构
    scale_factors = [0.97, 0.98, 0.99, 1, 1.01, 1.02, 1.03]
    energies = []
    lattice_constants = []

    # 从每个目录提取能量和晶格常数
    for scale in scale_factors:
        outcar_path = f"Scale_{scale}/OUTCAR"
        contcar_path = f"Scale_{scale}/CONTCAR"

        if os.path.exists(outcar_path) and os.path.exists(contcar_path):
            energy = extract_energy_from_outcar(outcar_path)
            lattice_constant = extract_lattice_constant_from_contcar(contcar_path)

            energies.append(energy)
            lattice_constants.append(lattice_constant)

            print(f"Scale = {scale}, Lattice constant = {lattice_constant:.4f} Å, Energy = {energy:.6f} eV")
        else:
            energies.append(np.nan)
            lattice_constants.append(np.nan)
            print(f"Scale = {scale}, data not found")

    # 过滤掉缺失的数据
    valid_indices = ~np.isnan(energies)
    scale_factors = np.array(scale_factors)[valid_indices]
    energies = np.array(energies)[valid_indices]
    lattice_constants = np.array(lattice_constants)[valid_indices]

    # 绘制能量随晶格常数变化的曲线
    plt.figure(figsize=(10, 6))
    plt.plot(lattice_constants, energies, 'o-', linewidth=2, markersize=8, color='green')
    plt.xlabel('Lattice Constant (Å)')
    plt.ylabel('Total Energy (eV)')
    plt.title('Lattice Constant Optimization for Graphene')
    plt.grid(True, alpha=0.3)

    # 拟合能量曲线以找到最优晶格常数
    if len(energies) > 3:
        # 使用二次多项式拟合
        coeffs = np.polyfit(lattice_constants, energies, 2)
        poly_fn = np.poly1d(coeffs)

        # 生成拟合曲线
        fit_lattice = np.linspace(min(lattice_constants), max(lattice_constants), 100)
        fit_energy = poly_fn(fit_lattice)

        plt.plot(fit_lattice, fit_energy, '--', color='orange', linewidth=2, label='Quadratic fit')

        # 找到能量最小值对应的晶格常数
        optimal_lattice = -coeffs[1] / (2 * coeffs[0])
        min_energy = poly_fn(optimal_lattice)

        plt.axvline(x=optimal_lattice, color='red', linestyle='--',
                    label=f'Optimal lattice constant = {optimal_lattice:.4f} Å')
        plt.legend()

        print(f"Optimal lattice constant: {optimal_lattice:.4f} Å")

    plt.tight_layout()
    plt.savefig('lattice_optimization.png', dpi=300)
    plt.show()

    return scale_factors, lattice_constants, energies


def main():
    """主函数"""
    print("Analyzing VASP calculation results...")
    print()


    # 分析晶格常数优化
    print("2. LATTICE CONSTANT OPTIMIZATION")
    print("-" * 40)
    scale_factors, lattice_constants, lattice_energies = analyze_lattice_constant_optimization()
    print()

if __name__ == "__main__":
    main()