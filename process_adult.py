# Create or update /content/medgan/process_adult.py

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

def process_adult_data(csv_file: Path, output_file: Path):
    """Process Adult dataset for medGAN."""
    print('Loading and processing Adult dataset')
    print(f'Reading file from: {csv_file}')
    
    # Read the CSV file
    data = pd.read_csv(csv_file)
    print(f'Successfully loaded data with shape: {data.shape}')
    
    # Convert all columns to numeric using LabelEncoder
    encoders = {}
    matrix = np.zeros((len(data), len(data.columns)), dtype=np.float32)
    
    for i, column in enumerate(data.columns):
        encoders[column] = LabelEncoder()
        matrix[:, i] = encoders[column].fit_transform(data[column].astype(str))
    
    # Normalize numeric columns
    numeric_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    for col in numeric_cols:
        col_idx = data.columns.get_loc(col)
        matrix[:, col_idx] = (matrix[:, col_idx] - matrix[:, col_idx].min()) / (matrix[:, col_idx].max() - matrix[:, col_idx].min())
    
    # Save outputs
    print(f'Saving outputs to {output_file}.*')
    with open(f'{output_file}.matrix', 'wb') as f:
        pickle.dump(matrix, f)
    with open(f'{output_file}.types', 'wb') as f:
        pickle.dump(encoders, f)
    
    print(f'Created matrix with shape: {matrix.shape}')
    print(f'Number of features: {len(data.columns)}')
    
    return matrix, encoders

if __name__ == '__main__':
    input_file = Path('/content/medgan/train_Adult_cleaned.csv')
    output_file = Path('/content/medgan/adult_processed')
    
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found at {input_file}")
        
    process_adult_data(input_file, output_file)