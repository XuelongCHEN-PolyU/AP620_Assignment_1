import numpy as np
import matplotlib.pyplot as plt
import os


def analyze_dirac_cone():
    """分析石墨烯狄拉克锥的轨道贡献"""
    # 读取能带数据
    band_data = np.loadtxt('REFORMATTED_BAND.dat')
    k_points = band_data[:, 0]
    energies = band_data[:, 1:]

    # 读取高对称点标签
    sym_points, labels = read_klabels('KLABELS')

    # 如果sym_points为空，则使用整个k点路径的中点
    if not sym_points:
        print("警告：KLABELS文件未提供有效的高对称点，将使用整个k点路径的中点作为K点位置")
        k_position = (k_points[0] + k_points[-1]) / 2
    else:
        # 找到K点位置
        k_index = None
        for i, label in enumerate(labels):
            # 扩展可能的K点标签识别
            if label.lower() in ['k', 'k$', 'k\\', 'k1', 'k2']:
                k_index = i
                break

        if k_index is None:
            print("警告：未找到K点标签，将使用路径中点作为近似")
            k_index = len(sym_points) // 2

        k_position = sym_points[k_index]

    # 找到最接近K点的k点索引
    k_index_band = np.argmin(np.abs(k_points - k_position))

    # 提取K点附近的能带数据
    window = 50  # 取K点前后各50个点
    start_idx = max(0, k_index_band - window)
    end_idx = min(len(k_points), k_index_band + window)

    k_window = k_points[start_idx:end_idx]
    e_window = energies[start_idx:end_idx, :]

    # 找出狄拉克锥对应的能带（能量最接近费米能级的两条能带）
    fermi_energy = 0  # VASPKIT已将费米能级设为0
    band_energies_at_k = energies[k_index_band, :]
    sorted_bands = np.argsort(np.abs(band_energies_at_k - fermi_energy))
    dirac_bands = sorted_bands[:2]  # 取最接近费米能级的两条能带

    print(f"狄拉克锥对应的能带索引: {dirac_bands}")

    # 绘制狄拉克锥附近的能带
    plt.figure(figsize=(10, 6))
    for band in dirac_bands:
        plt.plot(k_window, e_window[:, band], label=f'能带 {band + 1}')

    plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    plt.axvline(x=k_position, color='gray', linestyle='--', alpha=0.7)
    plt.title('石墨烯狄拉克锥附近的能带结构')
    plt.xlabel('k点路径')
    plt.ylabel('能量 (eV)')
    plt.legend()
    plt.savefig('dirac_cone.png', dpi=300)
    plt.show()

    # 分析轨道贡献
    analyze_orbital_contribution(dirac_bands)


def analyze_orbital_contribution(dirac_bands):
    """分析狄拉克锥的轨道贡献"""
    # 读取总态密度
    if os.path.exists(r"C:\Users\Chen Xuelong\Desktop\AP620\Graphene\Inputs\DOS\TDOS.dat"):
        tdos_data = np.loadtxt(r"C:\Users\Chen Xuelong\Desktop\AP620\Graphene\Inputs\DOS\TDOS.dat")
        energy_tdos = tdos_data[:, 0]
        tdos = tdos_data[:, 1]
    else:
        print("警告：未找到TDOS.dat文件")
        return

    # 读取分波态密度 - 从PDOS_USER文件读取
    orbital_contributions = {}
    orbital_labels = ['s轨道', 'px轨道', 'py轨道', 'pz轨道']

    if os.path.exists(r"C:\Users\Chen Xuelong\Desktop\AP620\Graphene\Inputs\DOS\PDOS_USER.dat"):
        try:
            # 读取PDOS_USER文件
            pdos_data = np.loadtxt(r"C:\Users\Chen Xuelong\Desktop\AP620\Graphene\Inputs\DOS\PDOS_USER.dat")

            # 假设文件格式：能量, s, px, py, pz
            energy_pdos = pdos_data[:, 0]
            orbital_contributions['s轨道'] = pdos_data[:, 1]
            orbital_contributions['px轨道'] = pdos_data[:, 2]
            orbital_contributions['py轨道'] = pdos_data[:, 3]
            orbital_contributions['pz轨道'] = pdos_data[:, 4]

            # 计算p轨道总和
            p_total = pdos_data[:, 2] + pdos_data[:, 3] + pdos_data[:, 4]
            orbital_contributions['p轨道'] = p_total

            # 添加p轨道标签
            orbital_labels.append('p轨道')

            print("成功从PDOS_USER文件读取轨道数据")
        except Exception as e:
            print(f"读取PDOS_USER文件时出错: {e}")
            return
    else:
        print("警告：未找到PDOS_USER文件")
        return

    # 计算狄拉克锥能量范围内的贡献
    energy_range = (-1, 1)  # 狄拉克锥附近的能量范围
    mask = (energy_tdos >= energy_range[0]) & (energy_tdos <= energy_range[1])

    total_integral = np.trapz(tdos[mask], energy_tdos[mask])
    print(f"\n狄拉克锥附近({energy_range[0]}到{energy_range[1]} eV)的态密度积分: {total_integral:.4f}")

    # 计算各轨道的贡献比例
    print("\n各轨道在狄拉克锥附近的贡献比例:")
    for label in orbital_labels:
        if label in orbital_contributions:
            pdos = orbital_contributions[label]
            integral = np.trapz(pdos[mask], energy_tdos[mask])
            percentage = (integral / total_integral) * 100
            print(f"{label}: {percentage:.2f}%")

    # 绘制态密度图
    plt.figure(figsize=(10, 6))
    plt.plot(energy_tdos, tdos, 'k-', label='总态密度')

    for label in orbital_labels:
        if label in orbital_contributions:
            plt.plot(energy_tdos, orbital_contributions[label], label=label)

    plt.axvline(x=0, color='r', linestyle='--', alpha=0.7)
    plt.axvspan(energy_range[0], energy_range[1], color='gray', alpha=0.2)
    plt.title('石墨烯态密度及各轨道贡献')
    plt.xlabel('能量 (eV)')
    plt.ylabel('态密度 (states/eV)')
    plt.xlim(-5, 5)
    plt.legend()
    plt.savefig('dos_contributions.png', dpi=300)
    plt.show()


def calculate_work_function():
    """计算单层石墨烯的功函数"""
    # 功函数 = 真空能级 - 费米能级
    # 在VASP计算中，真空能级可以从LOCPOT文件获取
    # 但如果没有LOCPOT文件，我们可以使用文献值或估算

    # 尝试从OUTCAR获取真空能级
    vacuum_level = None
    if os.path.exists(r"C:\Users\Chen Xuelong\Desktop\AP620\Graphene\Inputs\DOS\OUTCAR"):
        with open(r"C:\Users\Chen Xuelong\Desktop\AP620\Graphene\Inputs\DOS\OUTCAR") as f:
            for line in f:
                if 'vacuum level' in line:
                    try:
                        vacuum_level = float(line.split()[-1])
                        break
                    except:
                        continue

    # 如果找不到真空能级，使用石墨烯的典型值
    if vacuum_level is None:
        print("警告：未在OUTCAR中找到真空能级，使用石墨烯典型值4.5 eV")
        vacuum_level = 4.5

    # 费米能级在VASPKIT处理的数据中已设为0
    fermi_level = 0

    work_function = vacuum_level - fermi_level
    print(f"\n单层石墨烯的功函数: {work_function:.3f} eV")

    return work_function


def read_klabels(filename):
    """读取KLABELS文件 - 针对您提供的格式进行了修改"""
    sym_points = []
    labels = []

    if not os.path.exists(filename):
        print(f"警告：未找到{filename}文件")
        return sym_points, labels

    try:
        with open(filename, 'r') as f:
            lines = f.readlines()

            # 跳过标题行
            start_index = 0
            if lines and ("K-Label" in lines[0] or "Coordinate" in lines[0]):
                start_index = 1

            # 跳过可能存在的空行
            if start_index < len(lines) and not lines[start_index].strip():
                start_index += 1

            # 处理数据行
            for line in lines[start_index:]:
                if not line.strip():
                    continue

                parts = line.split()
                # 根据您提供的格式，标签在左侧，坐标在右侧
                if len(parts) >= 2:
                    try:
                        # 标签是第一个元素
                        label = parts[0].replace('\\', '')
                        # 坐标是最后一个元素
                        coord = float(parts[-1])

                        labels.append(label)
                        sym_points.append(coord)
                    except ValueError:
                        continue
    except Exception as e:
        print(f"读取KLABELS文件时出错: {e}")

    return sym_points, labels


if __name__ == "__main__":
    print("=" * 50)
    print("石墨烯狄拉克锥分析与功函数计算")
    print("=" * 50)

    try:
        # 分析狄拉克锥轨道贡献
        analyze_dirac_cone()

        # 计算功函数
        work_function = calculate_work_function()

        print("\n分析完成！")
    except Exception as e:
        print(f"程序运行出错: {e}")
        print("请检查输入文件是否存在且格式正确")