
import numpy as np

def wave_dynamics(field, velocity, dx, dt, c, steps):
    for step in range(steps):
        laplacian = (
            -6 * field +
            np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) +
            np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1) +
            np.roll(field, 1, axis=2) + np.roll(field, -1, axis=2)
        ) / dx**2

        accel = c**2 * laplacian
        velocity += accel * dt
        field += velocity * dt

        if step % 50 == 0:
            print(f"Step {step} complete.")

    return field, velocity
