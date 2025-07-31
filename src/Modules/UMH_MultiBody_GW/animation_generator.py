
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio.v2 as imageio
#import imageio


def create_tensor_evolution_animation(tensor_series, file_path, title, component=(0, 0), z_slice=None,cleanup=True, vmin=0,vmax=1,dpi=300):
    """
    Creates an animated GIF of the time evolution of a tensor component.
    tensor_series: list of (3,3,x,y,z) tensors
    component: tuple (i, j) for the tensor component to visualize
    z_slice: which z-plane to extract (defaults to center)
    """
    frames = []
    i, j = component

    for t, tensor in enumerate(tensor_series):
        if z_slice is None:
            z_slice = tensor.shape[4] // 2

        data_slice = tensor[i, j, :, :, z_slice]
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(data_slice.T, origin='lower', cmap='seismic',vmin=vmin,vmax=vmax)
        ax.set_title(f"Component ({i}{j}), Step {t}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()

        filename=f"{file_path}_Frame_{t:03d}.png"
        plt.savefig(filename, dpi=dpi)
        frames.append(imageio.imread(filename))
        plt.close()

    imageio.mimsave(f"{file_path}.gif", frames, duration=0.1)

    if cleanup:
        # Clean up temporary frames
        for frame in frames:
            try:
                os.remove(filename)
            except:
                pass

def generate_animation(tensor_sequence, save_path='tensor_animation.gif', fps=10):
    '''
    Generates an animated GIF from a sequence of tensor slices.

    Parameters:
        tensor_sequence (List[np.ndarray]): List of 2D arrays representing tensor slices at each timestep.
        save_path (str): Path to save the resulting GIF.
        fps (int): Frames per second of the animation.
    '''
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    images = []
    for idx, tensor in enumerate(tensor_sequence):
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(tensor, origin='lower', cmap='viridis')
        plt.colorbar(im, ax=ax)
        ax.set_title(f'Tensor Slice - Step {idx}')
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)
        plt.close(fig)
    imageio.mimsave(save_path, images, fps=fps)
    print(f"Animation saved to {save_path}")



def generate_gif_from_npy(directory, pattern="strain_timestep_", output_filename="animation.gif", duration=0.1):
    """
    Generate a GIF animation from a series of .npy strain files.
    Each file should represent a 2D slice or projection of the simulation at a given timestep.
    """
    filenames = sorted([f for f in os.listdir(directory) if f.startswith(pattern) and f.endswith(".npy")])
    images = []

    for fname in filenames:
        data = np.load(os.path.join(directory, fname))
        fig, ax = plt.subplots()
        im = ax.imshow(data, cmap='viridis', origin='lower')
        plt.colorbar(im, ax=ax)
        plt.title(fname)
        fig.canvas.draw()

        # Convert canvas to image
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)
        plt.close(fig)

    imageio.mimsave(os.path.join(directory, output_filename), images, duration=duration)
    print(f"Saved animation: {os.path.join(directory, output_filename)}")
