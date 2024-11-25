import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance
import pickle

def convert_synthetic_data(synthetic_file, encoders_file, output_csv):
    # Load synthetic data
    synthetic_data = np.load(synthetic_file + '.npy')
    print(f"Loaded synthetic data shape: {synthetic_data.shape}")
    
    # Load encoders
    with open(encoders_file + '.types', 'rb') as f:
        encoders = pickle.load(f)
    print(f"Loaded encoders for columns: {list(encoders.keys())}")
    
    # Create DataFrame
    df = pd.DataFrame(synthetic_data, columns=list(encoders.keys()))
    
    # Define numeric and categorical columns
    numeric_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    
    # Load original data to get value ranges
    original_df = pd.read_csv('/content/medgan/train_Adult_cleaned.csv')
    
    # Convert each column back to original format
    for column, encoder in encoders.items():
        if column in numeric_cols:
            # For numeric columns, use the original data range
            orig_min = original_df[column].min()
            orig_max = original_df[column].max()
            # Scale the synthetic data to the original range
            df[column] = df[column] * (orig_max - orig_min) + orig_min
            df[column] = df[column].round().astype(int)
        else:
            # Convert categorical columns back to original labels
            df[column] = encoder.inverse_transform(df[column].round().astype(int))
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Saved synthetic data to {output_csv}")
    return df

# Load original data first (for comparison)
original_df = pd.read_csv('/content/medgan/train_Adult_cleaned.csv')
print("Original data shape:", original_df.shape)

# Convert synthetic data
synthetic_df = convert_synthetic_data(
    '/content/medgan/synthetic_data',
    '/content/medgan/adult_processed',
    '/content/medgan/synthetic_adult.csv'
)

# Plotting function
def plot_distributions(original_df, synthetic_df, columns):
    n_cols = 2
    n_rows = (len(columns) + 1) // 2
    
    plt.figure(figsize=(15, 5 * n_rows))
    
    for i, col in enumerate(columns, 1):
        plt.subplot(n_rows, n_cols, i)
        
        # Plot original data
        sns.kdeplot(data=original_df[col], label='Original', alpha=0.5)
        # Plot synthetic data
        sns.kdeplot(data=synthetic_df[col], label='Synthetic', alpha=0.5)
        
        plt.title(f'Distribution of {col}')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

# Plot numeric columns
numeric_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
plot_distributions(original_df, synthetic_df, numeric_cols)

# Print summary statistics
print("\nNumeric Columns Statistics:")
print("\nOriginal Data:")
print(original_df[numeric_cols].describe())
print("\nSynthetic Data:")
print(synthetic_df[numeric_cols].describe())

# Compare categorical distributions
categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 
                   'relationship', 'race', 'sex', 'native-country']

print("\nCategorical Columns Distribution (Top 3 categories):")
for col in categorical_cols:
    print(f"\n{col}:")
    print("Original:")
    print(original_df[col].value_counts(normalize=True).head(3))
    print("\nSynthetic:")
    print(synthetic_df[col].value_counts(normalize=True).head(3))

# Calculate privacy metrics
def calculate_privacy_metrics(original_df, synthetic_df, numeric_cols):
    metrics = {}
    for col in numeric_cols:
        # Normalize the columns for fair comparison
        orig_norm = (original_df[col] - original_df[col].min()) / (original_df[col].max() - original_df[col].min())
        synth_norm = (synthetic_df[col] - synthetic_df[col].min()) / (synthetic_df[col].max() - synthetic_df[col].min())
        metrics[col] = wasserstein_distance(orig_norm, synth_norm)
    return metrics

privacy_metrics = calculate_privacy_metrics(original_df, synthetic_df, numeric_cols)
print("\nPrivacy Metrics (Wasserstein Distance):")
for col, score in privacy_metrics.items():
    print(f"{col}: {score:.4f}")

# Save detailed analysis
with open('/content/medgan/analysis_results.txt', 'w') as f:
    f.write("MedGAN Analysis Results\n")
    f.write("=======================\n\n")
    f.write(f"Number of synthetic samples generated: {len(synthetic_df)}\n")
    f.write(f"Number of features: {len(synthetic_df.columns)}\n\n")
    
    f.write("Numeric Columns Statistics:\n")
    f.write("\nOriginal Data:\n")
    f.write(original_df[numeric_cols].describe().to_string())
    f.write("\n\nSynthetic Data:\n")
    f.write(synthetic_df[numeric_cols].describe().to_string())
    
    f.write("\n\nPrivacy Metrics:\n")
    for col, score in privacy_metrics.items():
        f.write(f"{col}: {score:.4f}\n")

# Download results
from google.colab import files
files.download('/content/medgan/synthetic_adult.csv')
files.download('/content/medgan/analysis_results.txt')