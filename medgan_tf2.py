
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pickle

class MedganTF2:
    def __init__(self, input_dim, embedding_dim=128, noise_dim=128, 
                 generator_dims=(128, 128), discriminator_dims=(256, 128, 1),
                 compress_dims=(), decompress_dims=(), 
                 batch_norm_decay=0.99, l2_scale=0.001):
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.noise_dim = noise_dim
        self.generator_dims = list(generator_dims) + [embedding_dim]
        self.discriminator_dims = discriminator_dims
        self.compress_dims = list(compress_dims) + [embedding_dim]
        self.decompress_dims = list(decompress_dims) + [input_dim]
        self.batch_norm_decay = batch_norm_decay
        self.l2_scale = l2_scale
        
        # Initialize models
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        
        # Initialize optimizers
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.autoencoder_optimizer = tf.keras.optimizers.Adam(1e-4)

    def build_generator(self):
        model = tf.keras.Sequential()
        
        for dim in self.generator_dims[:-1]:
            model.add(tf.keras.layers.Dense(dim))
            model.add(tf.keras.layers.BatchNormalization(momentum=self.batch_norm_decay))
            model.add(tf.keras.layers.ReLU())
            
        model.add(tf.keras.layers.Dense(self.generator_dims[-1], activation='tanh'))
        return model
    
    def build_discriminator(self):
        model = tf.keras.Sequential()
        
        for dim in self.discriminator_dims[:-1]:
            model.add(tf.keras.layers.Dense(dim))
            model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
            model.add(tf.keras.layers.Dropout(0.3))
            
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        return model
    
    def build_encoder(self):
        model = tf.keras.Sequential()
        
        for dim in self.compress_dims:
            model.add(tf.keras.layers.Dense(dim))
            model.add(tf.keras.layers.ReLU())
            
        return model
    
    def build_decoder(self):
        model = tf.keras.Sequential()
        
        for dim in self.decompress_dims[:-1]:
            model.add(tf.keras.layers.Dense(dim))
            model.add(tf.keras.layers.ReLU())
            
        model.add(tf.keras.layers.Dense(self.decompress_dims[-1], activation='sigmoid'))
        return model

    @tf.function
    def train_step(self, real_data, noise):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate fake data
            generated_data = self.generator(noise, training=True)
            decoded_data = self.decoder(generated_data)
            
            # Get discriminator outputs
            real_output = self.discriminator(real_data, training=True)
            fake_output = self.discriminator(decoded_data, training=True)
            
            # Calculate losses
            gen_loss = tf.reduce_mean(tf.math.log(1e-10 + 1 - fake_output))
            disc_loss = -tf.reduce_mean(tf.math.log(1e-10 + real_output) + 
                                      tf.math.log(1e-10 + 1 - fake_output))

        # Calculate gradients and update weights
        gen_gradients = gen_tape.gradient(gen_loss, 
                                        self.generator.trainable_variables + 
                                        self.decoder.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, 
                                          self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(
            zip(gen_gradients, 
                self.generator.trainable_variables + 
                self.decoder.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(disc_gradients, self.discriminator.trainable_variables))

        return gen_loss, disc_loss

    def train(self, data_path, batch_size=100, epochs=100):
        # Load and preprocess data
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        train_data, val_data = train_test_split(data, test_size=0.1)
        
        # Training loop
        for epoch in range(epochs):
            gen_losses = []
            disc_losses = []
            
            # Train on batches
            for i in range(0, len(train_data), batch_size):
                batch_data = train_data[i:i + batch_size]
                noise = tf.random.normal([len(batch_data), self.noise_dim])
                
                gen_loss, disc_loss = self.train_step(batch_data, noise)
                gen_losses.append(gen_loss)
                disc_losses.append(disc_loss)
            
            # Print progress
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Gen Loss: {np.mean(gen_losses):.4f}, '
                      f'Disc Loss: {np.mean(disc_losses):.4f}')
    
    def generate(self, n_samples):
        noise = tf.random.normal([n_samples, self.noise_dim])
        generated = self.generator(noise)
        decoded = self.decoder(generated)
        return decoded.numpy()

# Training script
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file', type=str)
    parser.add_argument('output_file', type=str)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    
    # Load data to get input dimension
    with open(args.data_file, 'rb') as f:
        data = pickle.load(f)
    input_dim = data.shape[1]
    
    # Create and train model
    model = MedganTF2(input_dim=input_dim)
    model.train(args.data_file, batch_size=args.batch_size, epochs=args.epochs)
    
    # Generate synthetic data
    synthetic_data = model.generate(len(data))
    np.save(args.output_file, synthetic_data)