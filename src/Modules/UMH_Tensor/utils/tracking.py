import numpy as np
import matplotlib.pyplot as plt
import os
from numba import njit, prange

from scipy.optimize import curve_fit


def track_su2_fidelity(field, step_log, step):
    norms = np.sqrt(np.sum(field**2, axis=0))
    max_dev = np.max(np.abs(norms - 1.0))
    mean_dev = np.mean(np.abs(norms - 1.0))
    step_log.append((step, max_dev, mean_dev))

def plot_su2_deviation(log, file_path="su2_fidelity", title="su2 Fidelity",dpi=300):
    steps, max_dev, mean_dev = zip(*log)
    plt.plot(steps, max_dev, label="Max Deviation")
    plt.plot(steps, mean_dev, label="Mean Deviation")
    plt.xlabel("Step")
    plt.ylabel("Deviation from Unit Norm")
    plt.title(f"{title}: SU(2) Constraint Fidelity Over Time")
    plt.legend()
    plt.grid(True)
    filename=f"{file_path}_SU2_Deviation.png"
    plt.savefig(filename,dpi=dpi)
    plt.close()
    return filename

def track_energy_evolution(lattice, step_log, step):
    ke = np.sum(lattice.kinetic_energy())
    pe = np.sum(lattice.potential_energy())
    ne = np.sum(lattice.nonlinear_energy())
    total = ke + pe + ne
    step_log.append((step, ke, pe, ne, total))

def plot_energy_log(log, file_path="energy_evolution",title="Energy Evolution",dpi=300):
    steps, ke, pe, ne, total = zip(*log)
    plt.plot(steps, ke, label="KE")
    plt.plot(steps, pe, label="PE")
    plt.plot(steps, ne, label="NE")
    plt.plot(steps, total, label="Total")
    plt.xlabel("Step")
    plt.ylabel("Energy")
    plt.title(f"{title}: Real-Time Energy Evolution")
    plt.legend()
    plt.grid(True)
    filename=f"{file_path}_Energy_Evolution.png"
    plt.savefig(filename,dpi=dpi)
    plt.close()
    return filename

def log_local_energy_density(lattice, step, file_path, rank=0):
    local_energy = lattice.kinetic_energy() + lattice.potential_energy() + lattice.nonlinear_energy()
    np.save(f"{file_path}_Energy_Rank{rank}_Step_{step}.npy", local_energy)


    #Fermion



def plot_fermion_central2s_slice(lattice, file_path="fermion_field_slice", title="Fermion Field Slice",dpi=300):
    # Assume field shape is (C, X, Y, Z) or (C, N, N, N)
    # Extract central XY slice at mid-Z plane for component Ψ₁ (index 0)
    z_mid = lattice.field.shape[3] // 2  # index along Z
    field_slice = np.abs(lattice.field[0, :, :, z_mid])  # shape (128,128)

    plt.imshow(field_slice, cmap='viridis', origin='lower')
    plt.colorbar(label="|Ψ₁| (magnitude)")
    plt.title(f"{title} Field Central Slice (|Ψ₁|, Z-midplane)")
    filename=f"{file_path}_Field_Central_Slice.png"
    plt.savefig(filename,dpi=dpi)
    plt.close()
    return filename


def plot_field_snapshot(lattice, step, file_path="fermion_snapshot", title="Fermion SnapShot",dpi=300):
    z_mid = lattice.field.shape[3] // 2
    field_slice = np.abs(lattice.field[0, :, :, z_mid])  # |Ψ₁| central Z-plane

    plt.imshow(field_slice, cmap='viridis', origin='lower')
    plt.colorbar(label="|Ψ₁| (magnitude)")
    plt.title(f"{title} Field Slice |Ψ₁| at Step: {step:03}")
    filename=f"{file_path}_Step_{step:03}.png"
    plt.savefig(filename,dpi=dpi)
    plt.close()

    return filename





@njit(cache=True, parallel=True)
def compute_winding_number_density_njit(field):
    """
    Computes approximate topological (winding) density from SU(2) spinor field
    using unwrapped phase gradients of ψ₁.

    field shape: (2, X, Y, Z)
    Returns: (X, Y, Z) winding density (real scalar)
    """
    C, X, Y, Z = field.shape
    density = np.zeros((X, Y, Z), dtype=np.float64)

    for i in prange(X):
        for j in range(Y):
            for k in range(Z):
                # Get ψ₁ at current and neighbor points
                psi = field[0, i, j, k]
                psi_x = field[0, (i + 1) % X, j, k]
                psi_y = field[0, i, (j + 1) % Y, k]

                # Phase angles θ = atan2(imag, real)
                theta = np.arctan2(psi.imag, psi.real)
                theta_x = np.arctan2(psi_x.imag, psi_x.real)
                theta_y = np.arctan2(psi_y.imag, psi_y.real)

                # Angle differences
                dθx = theta_x - theta
                dθy = theta_y - theta

                # Wrap to [-π, π]
                if dθx > np.pi:
                    dθx -= 2 * np.pi
                elif dθx < -np.pi:
                    dθx += 2 * np.pi

                if dθy > np.pi:
                    dθy -= 2 * np.pi
                elif dθy < -np.pi:
                    dθy += 2 * np.pi

                # Approximate 2D winding: Jacobian-like product
                density[i, j, k] = dθx * dθy

    return density




def plot_topological_density(lattice,w_density, step=None, file_path="Fermion_Topo", title="Fermion Topo",dpi=300):
    print("Winding stats:", w_density.min(), w_density.max(), np.sum(w_density))

    # Take central Z-slice
    z_mid = w_density.shape[2] // 2
    slice2d = w_density[:, :, z_mid]

    plt.imshow(slice2d, cmap='seismic', origin='lower')
    plt.colorbar(label="Winding Density")
    title = f"{title}, Topological Charge Density (Z-slice)"
    if step is not None:
        title += f" at Step: {step}"
    plt.title(title)
    filename = f"{file_path}_Step_{step:03}.png" if step is not None else f"{file_path}.png"
    plt.savefig(filename,dpi=dpi)
    plt.close()

    return filename



def plot_total_winding(total_winding_history,file_path="Fermion_Topo", title="Fermion Topo", dpi=300):

    total_winding_history
    steps, winding = zip(*total_winding_history)

    plt.plot(steps, winding, marker='o')
    plt.title(f"{title}: Total Winding vs. Time")
    plt.xlabel("Step")
    plt.ylabel("Total Topological Charge")
    plt.grid(True)
    filename=f"{file_path}_Total_Winding_vs_Time.png"
    plt.savefig(filename,dpi=dpi)
    plt.close()

    return filename



def plot_spinor_phase_vector(lattice, step=None, file_path="Spinor_Phase_Vec", title="Spinor Phase Vec",dpi=300):
    # Assume field shape is (C, X, Y, Z)
    psi1 = lattice.field[0]  # first spinor component
    z_center = psi1.shape[2] // 2
    psi1_slice = psi1[:, :, z_center]

    complex_field = lattice.get_complex_spinors()
    ψ1 = complex_field[0]
    phase = np.angle(ψ1[:, :, z_center])


    # Compute phase
    #phase = np.angle(psi1_slice)

    # Create unit vectors from phase
    vx = np.cos(phase)
    vy = np.sin(phase)

    # Downsample for readability (optional)
    stride = 4
    X, Y = np.meshgrid(np.arange(0, psi1_slice.shape[1], stride),
                       np.arange(0, psi1_slice.shape[0], stride))
    U = vx[::stride, ::stride]
    V = vy[::stride, ::stride]

    plt.figure(figsize=(6, 6))
    plt.quiver(X, Y, U, V, scale=30, headwidth=2)
    title = f"{title}, Spinor Phase Vector Field (Z-midplane)"
    if step is not None:
        title += f" at Step: {step}"
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')
    plt.tight_layout()
    filename = f"{file_path}_Step_{step:03d}.png" if step is not None else f"{file_path}.png"
    plt.savefig(filename,dpi=dpi)
    plt.close()

    return filename


def plot_spinor_components(lattice, step=0, file_path="Fermion_Components", title="Fermion Components",dpi=300):
    """
    Plots real and imaginary parts of Ψ₁ and Ψ₂ at the central z-slice.
    Assumes field shape (C=2, X, Y, Z).
    """
    z_mid = lattice.field.shape[3] // 2

    re1 = lattice.field[0, :, :, z_mid]
    im1 = lattice.field[1, :, :, z_mid]
    re2 = lattice.field[2, :, :, z_mid]
    im2 = lattice.field[3, :, :, z_mid]

    print(f"{title}: Step: {step} - Im(Ψ₁): min={im1.min()}, max={im1.max()}")
    print(f"{title}: Step: {step} - Im(Ψ₂): min={im2.min()}, max={im2.max()}")

    components = [
        ("Re(Ψ₁)", re1),
        ("Im(Ψ₁)", im1),
        ("Re(Ψ₂)", re2),
        ("Im(Ψ₂)", im2),
    ]

    # 🔍 Debug print statements:
    print(f"{title}: Step: {step} - Im(Ψ₁): min={np.min(im1)}, max={np.max(im1)}")
    print(f"{title}: Step: {step} - Im(Ψ₂): min={np.min(im2)}, max={np.max(im2)}")

    components = [
        ("Re(Ψ₁)", re1),
        ("Im(Ψ₁)", im1),
        ("Re(Ψ₂)", re2),
        ("Im(Ψ₂)", im2),
    ]


    fig, axs = plt.subplots(2, 2, figsize=(10, 9))
    for ax, (label, data) in zip(axs.flat, components):
        im = ax.imshow(data, cmap="RdBu")
        ax.set_title(f"{title}: {label} at Step: {step}")
        fig.colorbar(im, ax=ax)


    plt.tight_layout()
    filename=f"{file_path}_Step_{step:03d}.png"
    plt.savefig(filename,dpi=dpi)
    plt.close()

    return filename



#Boson




def run_energy_decay_fit(energy_log, file_path="Boson_Energy_Decay_Fit", title="Boson Energy Decay Fit",dpi=300):
    """
    Fits total energy decay from boson simulation to an exponential model:
        E(t) = E0 * exp(-gamma * t) + E_inf
    and plots both the fit and residuals.

    Parameters:
        csv_path (str): Path to energy CSV file with columns: step, KE, PE, NE, total
        output_path (str): Output PNG path for plot
    """
    filename=f"{file_path}_Energy_Decay_Fit.png"

    try:
        steps, _, _, _, total_energy = zip(*energy_log)
        steps = np.array(steps)
        total_energy = np.array(total_energy)

        # Exponential decay model
        def exp_decay(t, E0, gamma, Einf):
            return E0 * np.exp(-gamma * t) + Einf


        # Fit the decay
        popt, _ = curve_fit(exp_decay, steps, total_energy, p0=(total_energy[0], 0.01, total_energy[-1]))
        E0, gamma, Einf = popt

        # Fitted curve and residuals
        fitted = exp_decay(steps, *popt)
        residuals = total_energy - fitted
        r_squared = 1 - np.sum(residuals**2) / np.sum((total_energy - np.mean(total_energy))**2)

        # Plot
        plt.figure(figsize=(10, 5))

        A, tau, C = popt
        eqn = f"E(t) = {A:.2f} · exp(-t / {tau:.2f}) + {C:.2f}"
        
        t_fit = np.linspace(steps.min(), steps.max(), 500)
        plt.plot(t_fit, exp_decay(t_fit, *popt), 'r-', label=eqn)


        plt.subplot(1, 2, 1)
        plt.plot(steps, total_energy, 'b.', label='Simulation')
        plt.plot(steps, fitted, 'r-', label='Exponential Fit')
        plt.xlabel("Step")
        plt.ylabel("Total Energy")
        plt.title(f"{title}: Energy Decay Fit\n$R^2 = {r_squared:.6f}$")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(steps, residuals, 'k')
        plt.xlabel("Step")
        plt.ylabel("Residuals")
        #plt.title("Fit Residuals")

        plt.tight_layout()
        
        plt.savefig(filename,dpi=dpi)
        plt.close()
        print(f"{title}: [decay_fit] Saved energy decay fit to: {filename}")
        print(f"{title}: [decay_fit] R² = {r_squared:.6f}, γ = {gamma:.6f}")

    except Exception as e:
        print(f"[decay_fit] Error in decay analysis: {e}")

    return filename

# Gravity Tensor


def plot_tensor_norm_from_data(tensor_list, steps=None, file_path="Gravity_Tensor_Norm.png", title="Gravity Tensor Norm", dpi=300):
    """
    Computes and plots Frobenius norm from a list of in-memory tensors.

    Parameters:
        tensor_list (List[np.ndarray]): Gravity tensor fields for each step.
        steps (List[int], optional): Matching step numbers (default: range(len(tensor_list))).
        output_path (str): Where to save the plot.
    """
    if steps is None:
        steps = list(range(len(tensor_list)))

    norms = [np.sqrt(np.sum(np.abs(tensor)**2)) for tensor in tensor_list]

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(steps, norms, marker='o')
    plt.xlabel("Step")
    plt.ylabel("Frobenius Norm ‖T‖")
    plt.title(f"{title}: Norm vs Time")
    plt.grid(True)
    plt.tight_layout()
    filename=f"{file_path}_Norm_vs_Time.png"
    plt.savefig(filename,dpi=dpi)
    plt.close()
    
    print(f"{title}: Saved Norm vs Time: {filename}")
    return steps, norms



def visualize_tensor_vectors(tensor_field, background_field, step, component=("0", "1"), z_slice=None, file_path="Gravity_Tensor_Vectors", title="Gravity Tensor Vectors",dpi=300):
    """
    Creates a 2D quiver (vector) plot of selected tensor components over a background scalar field.

    Parameters:
        tensor_field (ndarray): Tensor field, shape (C, X, Y, Z)
        background_field (ndarray): Scalar field, shape (X, Y, Z)
        step (int): Current simulation step
        component (tuple): Tuple of channel indices as strings (e.g. ("0", "1"))
        z_slice (int or None): Z slice index; defaults to midplane
        output_dir (str): Directory to save output image
        prefix (str): Filename prefix
    """

    C, X, Y, Z = tensor_field.shape
    if z_slice is None:
        z_slice = Z // 2

    # Parse component indices
    i1 = int(component[0])
    i2 = int(component[1])

    # Extract vector components and background slice
    U = tensor_field[i1, :, :, z_slice]
    V = tensor_field[i2, :, :, z_slice]
    bg_slice = background_field[:, :, z_slice]

    # Create meshgrid
    x = np.arange(U.shape[1])
    y = np.arange(U.shape[0])
    Xg, Yg = np.meshgrid(x, y)

    # Plot with quiver on background
    plt.figure(figsize=(6, 6))
    plt.imshow(bg_slice, cmap='Greys', origin='lower')
    plt.quiver(Xg, Yg, U, V, scale=1, scale_units='xy', angles='xy', width=0.003, color='cyan')
    plt.title(f"{title}: Tensor Vector Field — Components: {component[0]}, {component[1]} (Step: {step})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")

    filename = f"{file_path}_{component[0]}_{component[1]}_Step_{step:03d}.png"
    plt.savefig(filename,dpi=dpi)
    plt.close()
    print(f"{title}: Saved Vector Plot to: {filename}")
    return filename


#multibody_tensor



def plot_vector_quiver_with_norm(field, step, component=("0", "1"), z_slice=None, file_path="Vector_Quiver", title="Vector Quiver",dpi=300):
    """
    Plots a quiver (vector) plot of a 3D vector field's 2 components with a norm background.

    Parameters:
        field (ndarray): Array of shape (3, X, Y, Z) representing a 3D vector field.
        step (int or str): Step label for output naming.
        component (tuple): Which components to plot as vector arrows (e.g., ("0", "1")).
        z_slice (int): Which z-slice to visualize (default: center).
        output_path (str): Directory to save the output.
        prefix (str): File prefix for output image.
    """
    assert field.shape[0] == 3, f"{title}: Field must have 3 channels for a vector field"

    _, X, Y, Z = field.shape
    if z_slice is None:
        z_slice = Z // 2

    # Extract 2D slice of each requested component
    i1 = int(component[0])
    i2 = int(component[1])

    U = field[i1, :, :, z_slice]
    V = field[i2, :, :, z_slice]

    # Compute magnitude (norm) for background
    norm_slice = np.sqrt(np.sum(field[:, :, :, z_slice] ** 2, axis=0))

    # Grid for quiver
    x = np.arange(U.shape[1])
    y = np.arange(U.shape[0])
    Xg, Yg = np.meshgrid(x, y)

    # Plot
    plt.figure(figsize=(6, 6))
    bg = plt.imshow(norm_slice, cmap='gray', origin='lower')
    plt.colorbar(bg, label="||field||")

    plt.quiver(Xg, Yg, U, V, color='cyan', scale=1, scale_units='xy', angles='xy', width=0.003)
    plt.title(f"{title}: Vector Field Quiver — Components: {component[0]} & {component[1]} (Step: {step})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")

    filename = f"{file_path}_Vector_Field_Quiver_{component[0]}_{component[1]}_Step_{step}.png"
    plt.savefig(filename,dpi=dpi)
    plt.close()

    print(f"{title}: [vector_quiver_plot] Saved: {filename}")
    return filename


def plot_tensor_norm_vs_time(tensor_norm_history, file_path="Tensor_Norm_vs_Time.png", title="Tensor Norm vs Time", dpi=300):
    """
    Plots tensor Frobenius norm vs time (simulation step).

    Parameters:
        tensor_norm_history (list of tuples): [(step, norm), ...]
        output_path (str): Where to save the plot
    """

    steps, norms = zip(*tensor_norm_history)
    plt.figure(figsize=(8, 5))
    plt.plot(steps, norms, label="‖T‖ (Frobenius Norm)", color="darkblue")
    plt.xlabel("Simulation Step")
    plt.ylabel("Tensor Norm")
    plt.title(f"{title}: Norm vs Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    filename=f"{file_path}_Norm_vs_Time"
    plt.savefig(filename,dpi=dpi)
    plt.close()
    print(f"[tensor_norm_plot] Saved to {filename}")

    return filename



#Tensor Curvature


def visualize_tensor_component(field, step, component="0", z_slice=None, file_path="Tensor_Component", title="Tensor Component",dpi=300):
    shape = field.shape
    if len(shape) == 5:
        _, _, X, Y, Z = shape
        i, j = {"xx": (0, 0), "xy": (0, 1), "xz": (0, 2),
                "yx": (1, 0), "yy": (1, 1), "yz": (1, 2),
                "zx": (2, 0), "zy": (2, 1), "zz": (2, 2)}[component]
        data = field[i, j]
    elif len(shape) == 4:
        C, X, Y, Z = shape
        channel = int(component)
        data = field[channel]
    else:
        raise ValueError(f"Unsupported field shape: {shape}")

    if z_slice is None:
        z_slice = Z // 2

    slice_data = data[:, :, z_slice]

    plt.figure()
    img = plt.imshow(slice_data, cmap="inferno", origin="lower")
    plt.colorbar(img, label=f"Component {component}")
    plt.title(f"{title}: Step: {step} — Component {component}")
    filename = f"{file_path}_Step_{step:03d}_{component}.png"
    plt.savefig(filename,dpi=dpi)
    plt.close()
    print(f"{title}: Saved Component Plot: {filename}")
    return filename




def plot_tensor_norm_vs_time(norm_log, file_path="Curvature_Norm_vs_Time.png", title="Curvature Norm vs Time",dpi=300):
    """
    Plots Frobenius norm of curvature tensor vs. time.

    Parameters:
        norm_log (list of tuples): [(step, norm), ...]
        file_path (str): Filepath to save plot
    """

    steps, norms = zip(*norm_log)

    plt.figure(figsize=(8, 5))
    plt.plot(steps, norms, label="||Curvature Tensor||", color="blue", linewidth=2)
    plt.xlabel("Step")
    plt.ylabel("Frobenius Norm")
    plt.title(f"{title}: Norm vs Time")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    filename=f"{file_path}_Norm_vs_Time.png"
    plt.savefig(filename,dpi=dpi)
    plt.close()
    print(f"{title}: Saved Tensor vs Time: {filename}")

    return filename



#Thermodynamics


def plot_temperature_vs_time(energies, file_path="Temperature_vs_Time.png", title="Temperature_vs_Time",dpi=300):
    """
    Plot temperature vs time using thermo data.
    """
    #data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    #steps = np.arange(len(data))
    #temperature = data[:, 5]  # Assuming temperature is the 6th column
    step,ke,pe,ne,total,entropy,temperature = zip(*energies)

    plt.figure(figsize=(8, 5))
    plt.plot(step, temperature, color="red", label="Temperature")
    plt.xlabel("Sample Step")
    plt.ylabel("Temperature")
    plt.title(f"{title}: Temperature vs Time")
    plt.grid(True)
    plt.legend()
    filename=f"{file_path}_Temperature_vs_Time.png"
    plt.savefig(filename,dpi=dpi)
    plt.close()
    print(f"{title}: Saved temperature plot to: {filename}")

    return filename


def plot_entropy_vs_time(energies, file_path="Entropy_vs_Time", title="Entropy vs Time",dpi=300):
    """
    Plot temperature vs time using thermo data.
    """
    #data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    #steps = np.arange(len(data))
    #temperature = data[:, 5]  # Assuming temperature is the 6th column
    step,ke,pe,ne,total,entropy,temperature = zip(*energies)

    plt.figure(figsize=(8, 5))
    plt.plot(step, entropy, color="blue", label="Entropy")
    plt.xlabel("Sample Step")
    plt.ylabel("Entropy")
    plt.title(f"{title}: Entropy vs Time")
    plt.grid(True)
    plt.legend()
    filename=f"{file_path}_Entropy_vs_Time.png"
    plt.savefig(filename,dpi=dpi)
    plt.close()
    print(f"{title}: Saved entropy plot to: {filename}")

    return filename


def plot_energy_breakdown(energies, file_path="Energy_Breakdown.png", title="Energy Breakdown", dpi=300):
    """
    Plot temperature vs time using thermo data.
    """
    #data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    #steps = np.arange(len(data))
    #temperature = data[:, 5]  # Assuming temperature is the 6th column
    step,ke,pe,ne,total,entropy,temperature = zip(*energies)

    plt.figure(figsize=(8, 5))
    plt.plot(step, ke, color="red", label="Temperature")
    plt.plot(step, pe, color="yellow", label="Temperature")
    plt.plot(step, ne, color="blue", label="Temperature")
    plt.plot(step, total, color="green", label="Temperature")
    plt.xlabel("Sample Step")
    plt.ylabel("Energy")
    plt.title(f"{title}: Energy Breakdown")
    plt.grid(True)
    plt.legend()
    filename=f"{file_path}_Energy_Breakdown.png"
    plt.savefig(filename,dpi=dpi)
    plt.close()
    print(f"{title}: Saved energy plot to: {filename}")

    return filename