%%writefile /content/medgan/train_improved.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import argparse
from medgan_improved import ImprovedMedgan

def preprocess_data(data_path):
    """Preprocess the data with improved normalization."""
    # Load data
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Split into numeric and categorical columns
    numeric_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 
                   'capital-loss', 'hours-per-week']
    categorical_cols = ['workclass', 'education', 'marital-status', 'occupation',
                       'relationship', 'race', 'sex', 'native-country']
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Normalize numeric columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    # Add small noise to categorical variables
    epsilon = 1e-8
    for col in categorical_cols:
        df[col] = df[col] + epsilon * np.random.normal(size=len(df))
    
    return df.values, scaler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file', type=str)
    parser.add_argument('output_file', type=str)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--pretrain_epochs', type=int, default=100)
    args = parser.parse_args()
    
    # Preprocess data
    data, scaler = preprocess_data(args.data_file)
    
    # Create and train model
    model = ImprovedMedgan(
        input_dim=data.shape[1],
        embedding_dim=128,
        noise_dim=128,
        generator_dims=(256, 128),
        discriminator_dims=(256, 128, 64, 1),
        compress_dims=(128, 64),
        decompress_dims=(64, 128),
        learning_rate=0.0002,
        beta1=0.5
    )
    
    # Train the model
    model.train(
        data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        pretrain_epochs=args.pretrain_epochs
    )
    
    # Generate synthetic data
    synthetic_data = model.generate(len(data))
    
    # Save the results
    np.save(args.output_file, synthetic_data)
    with open(args.output_file + '_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

if __name__ == '__main__':
    main()