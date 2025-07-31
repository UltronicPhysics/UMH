import numpy as np
import os
import csv

def save_npy(file_path, array):
    np.save(f"{file_path}.npy", array)
    print(f"Saved NPY: {path}")

#def save_csv(name, array, outdir="output"):
#    os.makedirs(outdir, exist_ok=True)
#    path = os.path.join(outdir, name + ".csv")
#    np.savetxt(path, array, delimiter=",")
#    print(f"Saved CSV: {path}")


def save_csv(path, data_dict):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Metric", "Value"])
        for key, value in data_dict.items():
            writer.writerow([key, value])



def save_fig(name, plt, outdir="output", dpi=300):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name + ".png")
    plt.savefig(path, dpi=dpi)
    print(f"Saved Figure: {path}")
