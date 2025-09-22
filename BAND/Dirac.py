import numpy as np
import matplotlib.pyplot as plt
import os


def analyze_dirac_cone():
    """Analyze orbital contributions to graphene Dirac cone"""
    # Read band data
    band_data = np.loadtxt('REFORMATTED_BAND.dat')
    k_points = band_data[:, 0]
    energies = band_data[:, 1:]

    # Read high-symmetry point labels
    sym_points, labels = read_klabels('KLABELS')

    # If sym_points is empty, use midpoint of k-path
    if not sym_points:
        print("Warning: No valid high-symmetry points found in KLABELS file, using path midpoint as K-point position")
        k_position = (k_points[0] + k_points[-1]) / 2
    else:
        # Find K-point position
        k_index = None
        for i, label in enumerate(labels):
            # Extended recognition of possible K-point labels
            if label.lower() in ['k', 'k$', 'k\\', 'k1', 'k2']:
                k_index = i
                break

        if k_index is None:
            print("Warning: K-point label not found, using path midpoint as approximation")
            k_index = len(sym_points) // 2

        k_position = sym_points[k_index]

    # Find index of k-point closest to K-point
    k_index_band = np.argmin(np.abs(k_points - k_position))

    # Extract band data near K-point
    window = 50  # Take 50 points before and after K-point
    start_idx = max(0, k_index_band - window)
    end_idx = min(len(k_points), k_index_band + window)

    k_window = k_points[start_idx:end_idx]
    e_window = energies[start_idx:end_idx, :]

    # Identify Dirac cone bands (two bands closest to Fermi level)
    fermi_energy = 0  # VASPKIT sets Fermi level to 0
    band_energies_at_k = energies[k_index_band, :]
    sorted_bands = np.argsort(np.abs(band_energies_at_k - fermi_energy))
    dirac_bands = sorted_bands[:2]  # Take two bands closest to Fermi level

    print(f"Dirac cone band indices: {dirac_bands}")

    # Plot bands near Dirac cone
    plt.figure(figsize=(10, 6))
    for band in dirac_bands:
        plt.plot(k_window, e_window[:, band], label=f'Band {band + 1}')

    plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    plt.axvline(x=k_position, color='gray', linestyle='--', alpha=0.7)
    plt.title('Graphene band structure near Dirac cone')
    plt.xlabel('k-path')
    plt.ylabel('Energy (eV)')
    plt.legend()
    plt.savefig('dirac_cone.png', dpi=300)
    plt.show()

    # Analyze orbital contributions
    analyze_orbital_contribution(dirac_bands)


def analyze_orbital_contribution(dirac_bands):
    """Analyze orbital contributions to Dirac cone"""
    # Read total density of states
    if os.path.exists(r"C:\Users\IT\Desktop\Graphene\Inputs\DOS\TDOS.dat"):
        tdos_data = np.loadtxt(r"C:\Users\IT\Desktop\Graphene\Inputs\DOS\TDOS.dat")
        energy_tdos = tdos_data[:, 0]
        tdos = tdos_data[:, 1]
    else:
        print("Warning: TDOS.dat file not found")
        return

    # Read projected density of states - from PDOS_USER file
    orbital_contributions = {}
    orbital_labels = ['s-orbital', 'px-orbital', 'py-orbital', 'pz-orbital']

    if os.path.exists(r"C:\Users\IT\Desktop\Graphene\Inputs\DOS\PDOS_USER.dat"):
        try:
            # Read PDOS_USER file
            pdos_data = np.loadtxt(r"C:\Users\IT\Desktop\Graphene\Inputs\DOS\PDOS_USER.dat")

            # Assume file format: energy, s, px, py, pz
            energy_pdos = pdos_data[:, 0]
            orbital_contributions['s-orbital'] = pdos_data[:, 1]
            orbital_contributions['px-orbital'] = pdos_data[:, 2]
            orbital_contributions['py-orbital'] = pdos_data[:, 3]
            orbital_contributions['pz-orbital'] = pdos_data[:, 4]

            # Calculate total p-orbital contribution
            p_total = pdos_data[:, 2] + pdos_data[:, 3] + pdos_data[:, 4]
            orbital_contributions['p-orbital'] = p_total

            # Add p-orbital label
            orbital_labels.append('p-orbital')

            print("Successfully read orbital data from PDOS_USER file")
        except Exception as e:
            print(f"Error reading PDOS_USER file: {e}")
            return
    else:
        print("Warning: PDOS_USER file not found")
        return

    # Calculate contributions in energy range of Dirac cone
    energy_range = (-1, 1)  # Energy range near Dirac cone
    mask = (energy_tdos >= energy_range[0]) & (energy_tdos <= energy_range[1])

    total_integral = np.trapz(tdos[mask], energy_tdos[mask])
    print(f"\nDensity of states integral near Dirac cone ({energy_range[0]} to {energy_range[1]} eV): {total_integral:.4f}")

    # Calculate contribution percentages for each orbital
    print("\nOrbital contribution percentages near Dirac cone:")
    for label in orbital_labels:
        if label in orbital_contributions:
            pdos = orbital_contributions[label]
            integral = np.trapz(pdos[mask], energy_tdos[mask])
            percentage = (integral / total_integral) * 100
            print(f"{label}: {percentage:.2f}%")

    # Plot density of states
    plt.figure(figsize=(10, 6))
    plt.plot(energy_tdos, tdos, 'k-', label='Total DOS')

    for label in orbital_labels:
        if label in orbital_contributions:
            plt.plot(energy_tdos, orbital_contributions[label], label=label)

    plt.axvline(x=0, color='r', linestyle='--', alpha=0.7)
    plt.axvspan(energy_range[0], energy_range[1], color='gray', alpha=0.2)
    plt.title('Graphene density of states and orbital contributions')
    plt.xlabel('Energy (eV)')
    plt.ylabel('Density of states (states/eV)')
    plt.xlim(-5, 5)
    plt.legend()
    plt.savefig('dos_contributions.png', dpi=300)
    plt.show()

def read_klabels(filename):
    """Read KLABELS file - modified for the provided format"""
    sym_points = []
    labels = []

    if not os.path.exists(filename):
        print(f"Warning: {filename} file not found")
        return sym_points, labels

    try:
        with open(filename, 'r') as f:
            lines = f.readlines()

            # Skip header lines
            start_index = 0
            if lines and ("K-Label" in lines[0] or "Coordinate" in lines[0]):
                start_index = 1

            # Skip possible empty lines
            if start_index < len(lines) and not lines[start_index].strip():
                start_index += 1

            # Process data lines
            for line in lines[start_index:]:
                if not line.strip():
                    continue

                parts = line.split()
                # According to the provided format, labels are on the left, coordinates on the right
                if len(parts) >= 2:
                    try:
                        # Label is the first element
                        label = parts[0].replace('\\', '')
                        # Coordinate is the last element
                        coord = float(parts[-1])

                        labels.append(label)
                        sym_points.append(coord)
                    except ValueError:
                        continue
    except Exception as e:
        print(f"Error reading KLABELS file: {e}")

    return sym_points, labels


if __name__ == "__main__":
    print("=" * 50)
    print("Graphene Dirac Cone Analysis and Work Function Calculation")
    print("=" * 50)

    try:
        # Analyze Dirac cone orbital contributions
        analyze_dirac_cone()

        print("\nAnalysis completed!")
    except Exception as e:
        print(f"Error during execution: {e}")
        print("Please check if input files exist and have correct format")