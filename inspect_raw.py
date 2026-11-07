import pandas as pd
import numpy as np

df = pd.read_csv('data/raw.csv', header=0, low_memory=False)
if pd.isna(df.iloc[0, 0]) or str(df.iloc[0, 0]).strip() == '':
    df = df.iloc[1:].reset_index(drop=True)

date_cols = list(df.columns[1:-1])
for col in date_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

non_nulls = df[date_cols].notna().sum(axis=1)
print(f"Total meters: {len(df)}")
print(f"Meters with >= 330 days: {(non_nulls >= 330).sum()}")
print(f"Meters with 365 days: {(non_nulls == 365).sum()}")
print("CHK_STATE distribution:")
print(df['CHK_STATE'].value_counts())
