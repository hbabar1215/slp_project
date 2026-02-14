import pandas as pd 
import numpy as np

df = pd.read_csv('data/raw/labels.csv')

# Inspect data first
print(df.head())

# Create eligibility flag
df["Eligible"] = np.where(
    (df["Q1"] == 1) &  # consent given
    (df[["Q2", "Q3", "Q4", "Q5"]] == 1).all(axis=1),
    "Yes",
    "No"
)

# Check results
print(df["Eligible"].value_counts())