
import numpy as np
import matplotlib.pyplot as plt
from lattice import Lattice
from soliton import create_soliton
from tensor_tools import compute_strain_tensor, compute_ricci_tensor, compute_strain_scalar
from radial_analysis_3d import radial_profile_3d

size = 300
lattice = Lattice(size=size, tension=10.0, density=1.0, damping=0.005)
center = (size//2, size//2, size//2)
create_soliton(lattice, center=center, radius=8, amplitude=1.0)

timesteps = 300
dt = 0.01

for step in range(timesteps):
    lattice.step(dt)

strain_tensor = compute_strain_tensor(lattice.u)
ricci_tensor = compute_ricci_tensor(strain_tensor)
strain_scalar = compute_strain_scalar(strain_tensor)

radial_strain = radial_profile_3d(strain_scalar, center)
radial_ricci = radial_profile_3d(ricci_tensor, center)

plt.plot(radial_strain, label="Strain Scalar")
plt.plot(radial_ricci, label="Ricci Curvature")
plt.title("Tensor Curvature vs. Distance")
plt.xlabel("Radius")
plt.ylabel("Value")
plt.legend()
plt.grid()
plt.show()
