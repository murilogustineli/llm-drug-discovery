import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("/home/ubuntu/data/heart_failure_clinical_records_dataset.csv")

# Display basic information
print("Dataset Shape:", df.shape)
print("\nDataset Information:")
print(df.info())

print("\nDescriptive Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# Explore categorical variables
print("\nCategorical Variables Distribution:")
categorical_cols = [
    "anaemia",
    "diabetes",
    "high_blood_pressure",
    "sex",
    "smoking",
    "DEATH_EVENT",
]
for col in categorical_cols:
    print(f"\n{col} distribution:")
    print(df[col].value_counts())
    print(f"{col} percentage:")
    print(df[col].value_counts(normalize=True) * 100)

# Explore numerical variables
print("\nNumerical Variables Distribution:")
numerical_cols = [
    "age",
    "creatinine_phosphokinase",
    "ejection_fraction",
    "platelets",
    "serum_creatinine",
    "serum_sodium",
    "time",
]

# Create a directory for plots
import os

os.makedirs("/home/ubuntu/plots", exist_ok=True)

# Plot histograms for numerical variables
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols):
    plt.subplot(3, 3, i + 1)
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.savefig("/home/ubuntu/plots/numerical_distributions.png")

# Plot correlation matrix
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.savefig("/home/ubuntu/plots/correlation_matrix.png")

# Explore relationship between variables and death event
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols):
    plt.subplot(3, 3, i + 1)
    sns.boxplot(x="DEATH_EVENT", y=col, data=df)
    plt.title(f"{col} vs Death Event")
plt.tight_layout()
plt.savefig("/home/ubuntu/plots/numerical_vs_death.png")

# Create a summary dataframe for potential trial eligibility criteria
print("\nPotential Trial Eligibility Criteria Ranges:")
criteria_summary = pd.DataFrame(
    {
        "Feature": numerical_cols,
        "Min": [df[col].min() for col in numerical_cols],
        "Max": [df[col].max() for col in numerical_cols],
        "Mean": [df[col].mean() for col in numerical_cols],
        "Median": [df[col].median() for col in numerical_cols],
        "Q1": [df[col].quantile(0.25) for col in numerical_cols],
        "Q3": [df[col].quantile(0.75) for col in numerical_cols],
    }
)
print(criteria_summary)

# Save the criteria summary
criteria_summary.to_csv("/home/ubuntu/data/criteria_summary.csv", index=False)

# Prepare data for modeling (to be used later)
X = df.drop("DEATH_EVENT", axis=1)
y = df["DEATH_EVENT"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save processed data for later use
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)
train_data.to_csv("/home/ubuntu/data/train_data.csv", index=False)
test_data.to_csv("/home/ubuntu/data/test_data.csv", index=False)

print("\nData preprocessing complete. Files saved to /home/ubuntu/data/")
print("Visualizations saved to /home/ubuntu/plots/")
