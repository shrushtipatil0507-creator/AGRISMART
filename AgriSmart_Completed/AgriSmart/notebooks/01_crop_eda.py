import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
df = pd.read_csv("data/raw/Crop_recommendation.csv")

# 2. Create folder to save visualisations
output_folder = "reports/eda"
os.makedirs(output_folder, exist_ok=True)

sns.set_theme(style="whitegrid")

# 3. Crop distribution
plt.figure(figsize=(14, 7))
sns.countplot(data=df, y="label", order=df["label"].value_counts().index)
plt.title("Distribution of Crop Classes")
plt.xlabel("Number of Records")
plt.ylabel("Crop")
plt.tight_layout()
plt.savefig(f"{output_folder}/crop_distribution.png")
plt.show()

# 4. NPK distribution across crops
fig, axes = plt.subplots(1, 3, figsize=(20, 9))

for ax, feature in zip(axes, ["N", "P", "K"]):
    sns.boxplot(data=df, x="label", y=feature, ax=ax)
    ax.set_title(f"{feature} Levels Across Crops")
    ax.tick_params(axis="x", rotation=90)

plt.tight_layout()
plt.savefig(f"{output_folder}/npk_analysis.png")
plt.show()

# 5. Environmental feature distributions
features = ["temperature", "humidity", "ph", "rainfall"]

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

for ax, feature in zip(axes.flatten(), features):
    sns.histplot(data=df, x=feature, kde=True, ax=ax)
    ax.set_title(f"Distribution of {feature}")

plt.tight_layout()
plt.savefig(f"{output_folder}/environmental_distributions.png")
plt.show()

# 6. Correlation heatmap
plt.figure(figsize=(10, 7))
correlation = df.select_dtypes(include="number").corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{output_folder}/correlation_heatmap.png")
plt.show()

print("EDA completed successfully!")
print("Visualisations saved in:", output_folder)