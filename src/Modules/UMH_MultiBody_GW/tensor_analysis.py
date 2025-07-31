import numpy as np
import os

def analyze_tensor_fields(einstein_tensor, ricci_tensor, riemann_tensor, file_path, title):
    results = {}

    # Compute traces
    einstein_trace = np.trace(einstein_tensor, axis1=0, axis2=1)
    ricci_trace = np.trace(ricci_tensor, axis1=0, axis2=1)
    #riemann_trace = np.trace(riemann_tensor.reshape(4, 4, 4, 4), axis1=2, axis2=3)
    riemann_trace = np.trace(riemann_tensor, axis1=2, axis2=3)  # shape will be (4, 4, Nx, Ny, Nz)

    # Save to files
    np.save(f"{file_path}_Einstein_Trace.npy", einstein_trace)
    np.save(f"{file_path}_Ricci_Trace.npy", ricci_trace)
    np.save(f"{file_path}_Riemann_Trace.npy", riemann_trace)

    #np.savetxt(os.path.join(output_dir, "einstein_trace.txt"), einstein_trace)
    #np.savetxt(os.path.join(output_dir, "ricci_trace.txt"), ricci_trace)
    #np.savetxt(os.path.join(output_dir, "riemann_trace.txt"), riemann_trace)

    #print(f"{title}: Tensor field traces saved to:", file_path)

    results["Ricci Trace Mean"] = np.mean(ricci_trace)
    results["Einstein Trace Mean"] = np.mean(einstein_trace)
    results["Ricci Max"] = np.max(ricci_tensor)
    results["Einstein Max"] = np.max(einstein_tensor)

    return results  # ← ← ← This is the crucial line

def compute_tensor_magnitude(tensor):
    """Compute Frobenius norm (magnitude) of a 3x3 tensor."""
    return np.sqrt(np.sum(tensor**2, axis=(-2, -1)))

def compare_tensor_fields(field1, field2):
    """Return element-wise difference and relative error between two tensor fields."""
    diff = field1 - field2
    magnitude1 = compute_tensor_magnitude(field1)
    magnitude2 = compute_tensor_magnitude(field2)
    relative_error = np.where(magnitude2 != 0, np.abs(magnitude1 - magnitude2) / magnitude2, 0)
    return diff, relative_error

def trace_tensor_field(tensor_field):
    """Compute the trace of each tensor in a 3D tensor field."""
    return np.trace(tensor_field, axis1=-2, axis2=-1)

def symmetrize_tensor_field(tensor_field):
    """Ensure each 3x3 tensor is symmetric."""
    return 0.5 * (tensor_field + np.transpose(tensor_field, axes=(0, 1, 2, 4, 3)))
