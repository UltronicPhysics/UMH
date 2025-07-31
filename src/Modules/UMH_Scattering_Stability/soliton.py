
import numpy as np

def create_soliton(lattice, center, radius, amplitude=1.0, phase=0.0):
    x, y, z = np.indices(lattice.u.shape)
    cx, cy, cz = center
    r = np.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2)
    mask = r <= radius
    wave = amplitude * np.exp(1j * (phase + r[mask]))
    lattice.u[mask] += wave
