"""Guide to read parquet files in Python using pandas"""

import pandas as pd

# Load the parquet file
df = pd.read_parquet('train-00000-of-00001.parquet')

# View the first few rows
print(df.head())

# View the columns
print(df.columns)
