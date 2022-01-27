import os

import pandas as pd

n = 10000

df = pd.read_csv('data/Bank/bank-additional-full.csv', sep=';')
df_sub = df.sample(n)

cwd = os.getcwd()
path = os.path.join(cwd, 'data', 'Bank', f'bank_{n}.csv')

df_sub.to_csv(path, sep=';')