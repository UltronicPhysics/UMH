import numpy as np
import numba
import os
import sys
import json
import csv
import matplotlib
matplotlib.use('Agg')  # Set non-GUI backend before importing pyplot
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import pandas as pd
import matplotlib.colors as mcolors

from numba import njit, prange
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure  # Make sure you have scikit-image installed
from concurrent.futures import ProcessPoolExecutor
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom
from scipy.ndimage import uniform_filter1d
import matplotlib.colors as mcolors
from scipy.ndimage import map_coordinates


try:
    from ..UMH_SupportModules.UMH_RicciTensor import compute_ricci_tensor_approx,compute_scalar_curvature,compute_einstein_tensor,compute_einstein_tensor_complete,compute_ricci_tensor_from_components
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from UMH_SupportModules.UMH_RicciTensor import compute_ricci_tensor_approx,compute_scalar_curvature,compute_einstein_tensor,compute_einstein_tensor_complete,compute_ricci_tensor_from_components


#Trim beginning and ending frames to ensure good data, and no outside influence of numerical buildup.
def trim_edges(data, trim_start=2, trim_end=2):
    return data[trim_start:-trim_end] if trim_end > 0 else data[trim_start:]

#Not Used, but can downsample for Iso 3d for speed.  Currently async so not a real issue.
def downsample(data, factor=4):
    return data[::factor, ::factor, ::factor]

def center_crop(slice_, crop_size=128):
    cx, cy = np.array(slice_.shape) // 2
    return slice_[cx - crop_size//2: cx + crop_size//2, cy - crop_size//2: cy + crop_size//2]

def zoom_slice_for_display(slice_, zoom_factor=4):
    return zoom(slice_, zoom=zoom_factor, order=1)  # Linear interpolation

def crop_field(field, thickness):
    return field[thickness:-thickness, thickness:-thickness, thickness:-thickness]


#Save ZSlice of wave.
def save_zslice_image(file_path, title, name, array, axis, dpi=300, cmap="seismic"):
    if axis == 'xy':
        slice_ = array[:, :, array.shape[2] // 2]
    elif axis == 'xz':
        slice_ = array[:, array.shape[1] // 2, :]
    elif axis == 'yz':
        slice_ = array[array.shape[0] // 2, :, :]
    else:
        raise ValueError("Axis must be 'xy', 'xz', or 'yz'")
        
    if np.iscomplexobj(slice_):
        slice_ = np.abs(slice_)


    slice_zoomed=slice_
    #slice_ctr = center_crop(slice_, crop_size=128)
    #slice_zoomed = zoom_slice_for_display(slice_ctr, zoom_factor=1.5) #2

    vmax = np.percentile(slice_zoomed, 85)
    #vmin = -vmax if np.min(slice_zoomed) < 0 else 0
    vmin = np.percentile(slice_zoomed, 5)

    #print(f"{title}: Slice Zoomed: max={np.max(slice_zoomed)}, min={np.min(slice_zoomed)}, mean={np.mean(slice_zoomed)}")

    plt.figure(figsize=(12, 12))
    plt.imshow(slice_zoomed, cmap=cmap, interpolation='nearest',vmin=vmin,vmax=vmax) #cmap='seismic', vmin=-0.02, vmax=0.02  norm=LogNorm(vmin=1e-5, vmax=1)
    plt.colorbar()
    plt.title(f"{title}: {name} slice {axis.upper()}")
    plt.tight_layout()

    filename=f"{file_path}_{name}_Slice_{axis}.png"
    plt.savefig(filename, dpi=dpi) #, bbox_inches='tight', transparent=True
    plt.close()


def compute_radial_profile(field, center):
    """Computes radial average of a 3D scalar field."""
    grid = np.indices(field.shape).T - center
    radius = np.linalg.norm(grid, axis=-1)
    radius = radius.flatten()
    values = field.flatten()
    # Bin by radius
    bins = np.linspace(0, radius.max(), num=100)
    bin_means = np.zeros_like(bins)
    for i in range(len(bins) - 1):
        mask = (radius >= bins[i]) & (radius < bins[i + 1])
        if np.any(mask):
            bin_means[i] = np.mean(np.abs(values[mask]))
    return bins[:-1], bin_means[:-1]


def radial_plot(file_path, title, name, einstein_G_zz, anmgifary=None, dpi=300):

    # Assuming `einstein_G_zz` is loaded as a 3D numpy array
    center = np.array(einstein_G_zz.shape) // 2
    radii, radial_energy = compute_radial_profile(einstein_G_zz, center)

    plt.figure()
    plt.loglog(radii, radial_energy, marker='o')
    plt.xlabel('Radius (r)')
    plt.ylabel('Mean $|G_{zz}|$')
    plt.title(f"{title}: {name}: Gravitational Wave Energy Flux Decay")
    plt.grid(True)
    plt.savefig(f"{file_path}_{name}.png", dpi=dpi)
    plt.close()

    if anmgifary is not None: anmgifary.append(imageio.imread(filename))



def get_iso_value(volume, percentile=95):
    flat_vals = volume[np.isfinite(volume)].flatten()
    return np.percentile(flat_vals, percentile)

#Plot 3d Iso Image.
def save_isosurface_3d(file_path, title, name, tensor_data, iso_value=0.1, dpi=300, cmap='twilight', color_field=None):
    print(f"{title}: save_isosurface_3d_colormap({file_path}_{name}_3d.png, iso={iso_value}), min: {tensor_data.min():.5f}, max: {tensor_data.max():.5f}")

    if np.max(tensor_data) < 1e-8:
        print(f"[Skipped] {file_path} - {name} - Max value too low for isosurface.")
        return

    # Extract isosurface
    verts, faces, normals, values = measure.marching_cubes(np.abs(tensor_data), level=iso_value, step_size=3) #step_size=1

    # --- Colormap selection ---
    if color_field is not None:
        # Interpolate color_field values onto the isosurface vertices
        # color_field must have same shape as tensor_data!
        interp_vals = map_coordinates(color_field, verts.T, order=1, mode='nearest')
        scalars = interp_vals
    else:
        # Default: Use z-height
        scalars = verts[:, 2]

    norm = plt.Normalize(vmin=np.min(scalars), vmax=np.max(scalars))
    cmapN = plt.get_cmap(cmap)
    face_colors = cmapN(norm(scalars[faces].mean(axis=1)))

    # Setup 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces], facecolors=face_colors, edgecolor='none')

    ax.add_collection3d(mesh)
    ax.set_xlim(0, tensor_data.shape[0])
    ax.set_ylim(0, tensor_data.shape[1])
    ax.set_zlim(0, tensor_data.shape[2])
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=30, azim=45)

    # White background
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    ax.set_title(f"{title}: {name} Isosurface: {iso_value}")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.tight_layout()
    plt.savefig(f"{file_path}_{name}_3d.png", dpi=dpi)
    plt.close()


def low_pass_filter_3d(data, cutoff_fraction=0.1):
    shape = data.shape
    data_fft = np.fft.fftn(data)
    kx = np.fft.fftfreq(shape[0])
    ky = np.fft.fftfreq(shape[1])
    kz = np.fft.fftfreq(shape[2])
    kx, ky, kz = np.meshgrid(kx, ky, kz, indexing='ij')
    k_mag = np.sqrt(kx**2 + ky**2 + kz**2)
    data_fft[k_mag > cutoff_fraction] = 0
    return np.real(np.fft.ifftn(data_fft))



def save_npy(name, array, outdir="output"):
    #os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, name + ".npy"), array)

def save_csv_slice(name, array, axis='z', outdir="output"):
    import pandas as pd
    import os

    #os.makedirs(outdir, exist_ok=True)

    if len(array.shape) == 3:
        if axis == 'z':
            array = array[:, :, array.shape[2] // 2]
        elif axis == 'y':
            array = array[:, array.shape[1] // 2, :]
        elif axis == 'x':
            array = array[array.shape[0] // 2, :, :]
        else:
            raise ValueError("Axis must be 'x', 'y', or 'z'.")

    df = pd.DataFrame(array)
    df.to_csv(os.path.join(outdir, f"{name}_slice_{axis}.csv"), index=False)

def save_csv_3d(name, array, outdir="output"):
    #os.makedirs(outdir, exist_ok=True)
    nx, ny, nz = array.shape
    flat = [(x, y, z, array[x, y, z]) for x in range(nx) for y in range(ny) for z in range(nz)]
    df = pd.DataFrame(flat, columns=["X", "Y", "Z", "Value"])
    df.to_csv(os.path.join(outdir, name + "_3D.csv"), index=False)



    
def Vacuum_Einstein_Tensor(file_path, title, psi1_vac, psi2_vac, psi3_vac, dx, cz, dpi=300):
    # --- Vacuum Einstein Tensor: Before any soliton/topology introduced ---
    G_vac = compute_einstein_tensor_complete(psi1_vac, psi2_vac, psi3_vac, dx=dx)
    G_vac_mag = np.linalg.norm(G_vac, axis=0)  # or use suitable contraction for your data
    plt.figure()
    plt.imshow(G_vac_mag[:, :, cz], cmap="viridis")
    plt.colorbar()
    plt.title(f"{title} Einstein Tensor |G| (Vacuum, central z-slice)")
    plt.tight_layout()
    plt.savefig(f"{file_path}_EinsteinTensor_Vacuum.png", dpi=dpi)
    plt.close()


def Soliton_Einstein_Tensor(file_path, title, G_soliton, cz, dpi=300): #psi1, psi2, psi3, dx
    # --- Soliton Einstein Tensor: Final configuration ---
    #G_soliton = compute_einstein_tensor_complete(psi1, psi2, psi3, dx=dx)
    G_soliton_mag = np.linalg.norm(G_soliton, axis=0)  # contract as appropriate for your tensor

    plt.figure()
    plt.imshow(G_soliton_mag[:, :, cz], cmap="viridis")
    plt.colorbar()
    plt.title(f"{title} Einstein Tensor |G| (Soliton, central z-slice)")
    plt.tight_layout()
    plt.savefig(f"{file_path}_EinsteinTensor_Soliton.png", dpi=dpi)
    plt.close()



def Stress_Energy_Tensor(file_path, title, T_soliton, G_soliton, cz, dpi=300):
    """
    Plots norm (Frobenius) overlays and residual for G and T, for central z-slice.
    Inputs:
        T_soliton: (3, 3, Nx, Ny, Nz) or (3, Nx, Ny, Nz)
        G_soliton: (3, 3, Nx, Ny, Nz) or (3, Nx, Ny, Nz)
        cz: index of central z-slice (int)
    """

    # Compute norms at each voxel
    if T_soliton.ndim == 5:  # full tensor
        T_norm = np.sqrt(np.sum(T_soliton ** 2, axis=(0, 1)))  # (Nx, Ny, Nz)
    elif T_soliton.ndim == 4:  # diagonal only
        T_norm = np.sqrt(np.sum(T_soliton ** 2, axis=0))  # (Nx, Ny, Nz)
    else:
        raise ValueError("Unexpected shape for T_soliton.")

    if G_soliton.ndim == 5:
        G_norm = np.sqrt(np.sum(G_soliton ** 2, axis=(0, 1)))
    elif G_soliton.ndim == 4:
        G_norm = np.sqrt(np.sum(G_soliton ** 2, axis=0))
    else:
        raise ValueError("Unexpected shape for G_soliton.")

    # Residual (norm difference at each voxel)
    residual = np.abs(G_norm - 8 * np.pi * T_norm)

    # Plot overlays
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(G_norm[:, :, cz], cmap="Blues", alpha=0.7)
    plt.imshow(8 * np.pi * T_norm[:, :, cz], cmap="Oranges", alpha=0.4)
    plt.colorbar()
    plt.title(f"{title}: $|G|$ (blue) and $8\\pi|T|$ (orange), Central z-slice")

    plt.subplot(1, 2, 2)
    plt.imshow(residual[:, :, cz], cmap="magma")
    plt.colorbar()
    plt.title(f"{title}: $|G - 8\\pi T|$ (Residual, Central Slice)")

    plt.tight_layout()
    plt.savefig(f"{file_path}_G_vs_T_Comparison.png", dpi=dpi)
    plt.close()



def EinsteinTensor_Stats(file_path, title, G_soliton, residual):

    if G_soliton.ndim == 5:
        G_soliton_mag = np.linalg.norm(G_soliton, axis=(0,1))
    elif G_soliton.ndim == 4:
        G_soliton_mag = np.linalg.norm(G_soliton, axis=0)
    else:
        raise ValueError("Unexpected shape for G_soliton.")

    mean_G = np.mean(G_soliton_mag)
    max_G = np.max(G_soliton_mag)
    mean_residual = np.mean(residual)
    max_residual = np.max(residual)
    print(f"{title}: Einstein Tensor (Final): mean={mean_G:.2e}, max={max_G:.2e}")
    print(f"{title}: Residual (|G - 8πT|): mean={mean_residual:.2e}, max={max_residual:.2e}")

    # Optionally, save as CSV for table in appendix:
    with open(f"{file_path}_EinsteinTensor_Stats.csv", "w") as f:
        f.write("Quantity,Mean,Max\n")
        f.write(f"G_soliton_mag,{mean_G},{max_G}\n")
        f.write(f"Residual,{mean_residual},{max_residual}\n")
