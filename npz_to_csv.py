import pandas as pd
import numpy as np

data = np.load('data/Synthetic/Synthetic.npz')
pd.DataFrame(data).to_csv("data/Synthetic/synthetic.csv")