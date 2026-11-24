import pandas as pd

# Load the crop recommendation dataset
df = pd.read_csv("data/raw/Crop_recommendation.csv")

# Display basic information
print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nCrop categories:")
print(df["label"].value_counts())
print("\nTotal duplicate rows:")
print(df.duplicated().sum())

print("\nData types:")
print(df.dtypes)

print("\nStatistical summary:")
print(df.describe())

# Additional dataset audit

print("\n========== DATASET AUDIT ==========")

# 1. Duplicate records
print("\nDuplicate rows:")
print(df.duplicated().sum())

# 2. Data types
print("\nData types:")
print(df.dtypes)

# 3. Crop distribution
print("\nCrop distribution:")
print(df["label"].value_counts().sort_index())

# 4. Unique crop count
print("\nNumber of unique crops:")
print(df["label"].nunique())

# 5. Statistical summary
print("\nStatistical summary:")
print(df.describe())

# 6. Check invalid numeric values
print("\nInfinite values:")
print(df.select_dtypes(include="number").isin(
    [float("inf"), float("-inf")]
).sum())