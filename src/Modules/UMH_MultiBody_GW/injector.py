
import numpy as np

def inject_solitons(field, positions, amplitudes, width=3.0):
    """Inject solitonic sources into the simulation field."""
    shape = field.shape
    for pos, amp in zip(positions, amplitudes):
        z0, y0, x0 = pos
        for z in range(max(0, z0 - 5), min(shape[0], z0 + 6)):
            for y in range(max(0, y0 - 5), min(shape[1], y0 + 6)):
                for x in range(max(0, x0 - 5), min(shape[2], x0 + 6)):
                    r2 = (z - z0)**2 + (y - y0)**2 + (x - x0)**2
                    field[z, y, x] += amp * np.exp(-r2 / (2 * width**2))
    return field
