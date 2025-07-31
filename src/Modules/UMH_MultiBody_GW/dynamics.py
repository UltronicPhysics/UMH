# dynamics.py
import numpy as np

def initialize_grid(shape, solitons=None):
    field = np.zeros(shape)
    if solitons:
        for s in solitons:
            insert_soliton(field, *s)
    return field

def insert_soliton(field, x, y, z, amplitude, radius):
    X, Y, Z = np.indices(field.shape)
    dist = np.sqrt((X - x)**2 + (Y - y)**2 + (Z - z)**2)
    field += amplitude * np.exp(-(dist / radius)**2)

def update_field(field, dt=0.1, damping=0.01):
    laplacian = (
        np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) +
        np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1) +
        np.roll(field, 1, axis=2) + np.roll(field, -1, axis=2) -
        6 * field
    )
    field += dt * (laplacian - damping * field)
    return field
