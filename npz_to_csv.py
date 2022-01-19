import numpy as np

data = np.load('data/Synthetic/Synthetic.npz')
for key, value in data.items():
    np.savetxt("." + key + ".csv", value)