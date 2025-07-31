
import numpy as np
import matplotlib.pyplot as plt
import os

def generate_animation_frames(field_history, axis, outdir="output/animation_frames"):
    os.makedirs(outdir, exist_ok=True)
    for i, field in enumerate(field_history):
        if axis == 'xy':
            slice_ = field[:, :, field.shape[2] // 2]
        elif axis == 'xz':
            slice_ = field[:, field.shape[1] // 2, :]
        elif axis == 'yz':
            slice_ = field[field.shape[0] // 2, :, :]
        else:
            raise ValueError("Axis must be 'xy', 'xz', or 'yz'")

        plt.figure(figsize=(6, 6))
        plt.imshow(slice_, cmap='viridis')
        plt.title(f"Frame {i}")
        plt.colorbar()
        frame_path = os.path.join(outdir, f"frame_{i:04d}.png")
        plt.savefig(frame_path)
        plt.close()
