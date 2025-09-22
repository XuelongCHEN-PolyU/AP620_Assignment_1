import numpy as np
import matplotlib.pyplot as plt
from pymatgen.io.vasp.outputs import Outcar
from pymatgen.io.vasp import Locpot
import matplotlib
import os


# Set up Chinese font support
def setup_chinese_font():
    """
    Configure Chinese font display to resolve Chinese character rendering issues
    """
    # Set font path based on operating system
    if os.name == 'nt':  # Windows system
        plt.rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei font
    elif os.name == 'posix':  # macOS/Linux systems
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']  # Use WenQuanYi Micro Hei font

    plt.rcParams['axes.unicode_minus'] = False  # Resolve negative sign display issues
    print("Chinese font configuration completed")


# 1. Read OUTCAR file to obtain Fermi energy level
def get_fermi_energy(outcar_file='OUTCAR'):
    """
    Read Fermi energy level from OUTCAR file
    """
    outcar = Outcar(outcar_file)
    fermi_energy = outcar.efermi
    print(f"Fermi energy extracted from {outcar_file}: {fermi_energy:.6f} eV")
    return fermi_energy


# 2. Read LOCPOT file and calculate average electrostatic potential along the z-direction
def get_average_potential_along_z(locpot_file='LOCPOT'):
    """
    Read LOCPOT file and calculate the average electrostatic potential along the z-direction
    Returns z-coordinates and corresponding average electrostatic potential
    """
    locpot = Locpot.from_file(locpot_file)
    pot_data = locpot.data['total']  # Obtain 3D electrostatic potential data

    # Get grid dimensions
    ngx, ngy, ngz = pot_data.shape
    print(f"Electrostatic potential grid dimensions: ({ngx}, {ngy}, {ngz})")

    # Calculate xy-plane average to obtain 1D electrostatic potential along z-direction
    average_potential_z = np.mean(np.mean(pot_data, axis=0), axis=0)

    # Generate z-axis coordinates (normalized to 0-1 range)
    z_coords = np.linspace(0, 1, ngz)

    return z_coords, average_potential_z


# 3. Find vacuum level (average value in the vacuum region)
def find_vacuum_level(z_coords, potential, vacuum_region=[0.6, 0.9]):
    """
    Calculate the average electrostatic potential in the specified vacuum region as the vacuum level
    """
    # Determine index range for the vacuum region
    idx_start = np.argmin(np.abs(z_coords - vacuum_region[0]))
    idx_end = np.argmin(np.abs(z_coords - vacuum_region[1]))

    # Calculate average electrostatic potential in the vacuum region
    vacuum_potential = np.mean(potential[idx_start:idx_end])
    print(f"Average electrostatic potential in vacuum region {vacuum_region}: {vacuum_potential:.6f} eV")

    return vacuum_potential


# 4. Calculate and visualize work function
def calculate_work_function(outcar_file='OUTCAR', locpot_file='LOCPOT', vacuum_region=[0.6, 0.9]):
    """
    Main function: Calculate work function and plot electrostatic potential distribution
    """
    # Configure Chinese font
    setup_chinese_font()

    # Obtain Fermi energy level
    e_fermi = get_fermi_energy(outcar_file)

    # Obtain average electrostatic potential along z-direction
    z_coords, avg_potential = get_average_potential_along_z(locpot_file)

    # Find vacuum level
    vacuum_level = find_vacuum_level(z_coords, avg_potential, vacuum_region)

    # Calculate work function
    work_function = vacuum_level - e_fermi
    print(f"\nCalculation results:")
    print(f"Vacuum level: {vacuum_level:.6f} eV")
    print(f"Fermi energy level: {e_fermi:.6f} eV")
    print(f"Work function: {work_function:.6f} eV")

    # Plot electrostatic potential distribution
    plt.figure(figsize=(12, 7))
    plt.plot(z_coords, avg_potential, 'b-', linewidth=2, label='Average electrostatic potential')
    plt.axhline(y=vacuum_level, color='r', linestyle='--', label=f'Vacuum level ({vacuum_level:.4f} eV)')
    plt.axhline(y=e_fermi, color='g', linestyle='--', label=f'Fermi level ({e_fermi:.4f} eV)')

    # Mark vacuum region
    plt.axvspan(vacuum_region[0], vacuum_region[1], alpha=0.2, color='gray', label='Vacuum region')

    plt.xlabel('Z-coordinate (normalized)', fontsize=12)
    plt.ylabel('Electrostatic potential (eV)', fontsize=12)
    plt.title('Average Electrostatic Potential Distribution Along Z-direction and Work Function Calculation for Graphene', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)

    # Add work function value on the plot
    textstr = f'Work function = {work_function:.4f} eV'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig('graphene_work_function_calculation.png', dpi=300, bbox_inches='tight')
    plt.show()

    return work_function


# 5. Execute calculation
if __name__ == "__main__":
    # Default uses OUTCAR and LOCPOT files in the current directory
    # File paths and vacuum region range can be specified
    work_function = calculate_work_function(
        outcar_file='OUTCAR',
        locpot_file='LOCPOT',
        vacuum_region=[0.6, 0.9]  # Adjust vacuum region according to your system
    )