import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import json


def export_field_snapshot(field, step, file_path, title, component_label="|Ψ₁| (magnitude)", vmin=None, vmax=None,dpi=300):
    norm = np.linalg.norm(field, axis=0)
    z_slice = norm.shape[2] // 2

    img = plt.imshow(norm[:, :, z_slice] , cmap='inferno', origin='lower', vmin=vmin, vmax=vmax)  # or 'viridis', etc.
    plt.colorbar(img, label=component_label) #"|Ψ₁| (magnitude)"

    plt.title(f"{title}, Slice Step: {step} — Component: {component_label}")

    filename=f"{file_path}_Slice_Step_{step}.png"
    plt.savefig(filename,dpi=dpi)
    plt.close()

    return filename

def export_tensor_fields(*tensors, file_path):
    for i, tensor in enumerate(tensors):
        np.save(f"{file_path}_Tensor_{i}.npy", tensor)

def export_distribution_data(data, file_path):
    with open(f"{file_path}.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["KE", "PE", "NE"])
        writer.writerows(data)

def export_thermo_data(data, file_path):
    with open(f"{file_path}.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Step","KE", "PE", "NE", "Total", "Entropy", "Temperature"])
        writer.writerows(data)

def export_metadata(metadata, file_path="simulation_conditions"):
    # Save metadata for reproducibility  
    with open(f"{file_path}.json", "w") as f:
        json.dump(metadata, f, indent=4)


# Safe export wrapper
def export_field_raw_auto(lattice, step, file_path="field_raw"):
    try:
        field = lattice.get_complex_spinors()
    except (AttributeError, IndexError):
        field = lattice.field  # fallback to raw field if not spinor

    filename = f"{file_path}_Field_Raw_Step_{step:03d}.npy"
    np.save(filename, field.astype(np.complex64 if np.iscomplexobj(field) else np.float32))
    return filename

def export_field_raw_curvature(field, step, file_path="field_raw"):
    filename = f"{file_path}_Field_Raw_Step_{step:03d}.npy"
    np.save(filename, field.astype(np.complex64 if np.iscomplexobj(field) else np.float32))
    return filename
