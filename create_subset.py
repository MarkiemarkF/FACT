import os
import pandas as pd

"""
THIS FILE IS NOT IN USE!
USE THE convert_data_columns function in 'alter_data.py'!
"""

"""
Create a subset of 
"""

n = 10000

df = pd.read_csv('data/Bank/bank-additional-full.csv', sep=';')
df_sub = df.sample(n)

cwd = os.getcwd()
path = os.path.join(cwd, 'data', 'Bank', f'bank_{n}.csv')

df_sub.to_csv(path, sep=';')