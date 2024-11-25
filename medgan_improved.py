
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import pickle

class ImprovedMedgan:
    def __init__(self,
                 input_dim,
                 embedding_dim=128,
                 noise_dim=128,
                 generator_dims=(256, 128),
                 discriminator_dims=(256, 128, 64, 1),
                 compress_dims=(128, 64),
                 decompress_dims=(64, 128),
                 batch_norm_decay=0.9,
                 l2_scale=0.001,
                 learning_rate=0.0002,
                 beta1=0.5):
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.noise_dim = noise_dim
        self.generator_dims = generator_dims
        self.discriminator_dims = discriminator_dims
        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims
        self.batch_norm_decay = batch_norm_decay
        self.l2_scale = l2_scale
        self.learning_rate = learning_rate
        self.beta1 = beta1
        
        # Build models
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        
        # Build optimizers
        self.generator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate, beta_1=beta1)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate, beta_1=beta1)
        self.autoencoder_optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate, beta_1=beta1)

    def build_generator(self):
        noise_input = Input(shape=(self.noise_dim,))
        x = noise_input
        
        # Generator layers
        for dim in self.generator_dims:
            x = Dense(dim)(x)
            x = BatchNormalization(momentum=self.batch_norm_decay)(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = Dropout(0.3)(x)
        
        # Output layer
        x = Dense(self.embedding_dim, activation='tanh')(x)
        
        return Model(noise_input, x, name='generator')

    def build_discriminator(self):
        data_input = Input(shape=(self.input_dim,))
        x = data_input
        
        # Discriminator layers
        for dim in self.discriminator_dims[:-1]:
            x = Dense(dim)(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = Dropout(0.3)(x)
        
        # Output layer
        x = Dense(1, activation='sigmoid')(x)
        
        return Model(data_input, x, name='discriminator')

    def build_encoder(self):
        data_input = Input(shape=(self.input_dim,))
        x = data_input
        
        # Encoder layers
        for dim in self.compress_dims:
            x = Dense(dim)(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = Dropout(0.2)(x)
        
        # Output layer
        x = Dense(self.embedding_dim)(x)
        
        return Model(data_input, x, name='encoder')

    def build_decoder(self):
        embedding_input = Input(shape=(self.embedding_dim,))
        x = embedding_input
        
        # Decoder layers
        for dim in self.decompress_dims:
            x = Dense(dim)(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = Dropout(0.2)(x)
        
        # Output layer
        x = Dense(self.input_dim, activation='sigmoid')(x)
        
        return Model(embedding_input, x, name='decoder')

    @tf.function
    def train_step(self, real_data, batch_size):
        # Generate random noise
        noise = tf.random.normal([batch_size, self.noise_dim])
        
        with tf.GradientTape(persistent=True) as tape:
            # Generate fake data
            generated_embeddings = self.generator(noise, training=True)
            generated_data = self.decoder(generated_embeddings, training=True)
            
            # Get discriminator outputs
            real_output = self.discriminator(real_data, training=True)
            fake_output = self.discriminator(generated_data, training=True)
            
            # Calculate losses
            gen_loss = self.generator_loss(fake_output, real_data, generated_data)
            disc_loss = self.discriminator_loss(real_output, fake_output)
            
        # Calculate gradients
        gen_gradients = tape.gradient(
            gen_loss, 
            self.generator.trainable_variables + self.decoder.trainable_variables
        )
        disc_gradients = tape.gradient(
            disc_loss,
            self.discriminator.trainable_variables
        )
        
        # Apply gradients
        self.generator_optimizer.apply_gradients(
            zip(gen_gradients, 
                self.generator.trainable_variables + self.decoder.trainable_variables)
        )
        self.discriminator_optimizer.apply_gradients(
            zip(disc_gradients,
                self.discriminator.trainable_variables)
        )
        
        return gen_loss, disc_loss

    def generator_loss(self, fake_output, real_data, generated_data):
        # Traditional GAN loss
        gan_loss = tf.reduce_mean(tf.math.log(1e-10 + 1 - fake_output))
        
        # Feature matching loss
        feature_loss = tf.reduce_mean(
            tf.abs(tf.reduce_mean(real_data, axis=0) - 
                  tf.reduce_mean(generated_data, axis=0))
        )
        
        # Reconstruction loss
        reconstruction_loss = tf.reduce_mean(tf.square(real_data - generated_data))
        
        return gan_loss + 0.1 * feature_loss + 0.1 * reconstruction_loss

    def discriminator_loss(self, real_output, fake_output):
        real_loss = tf.reduce_mean(tf.math.log(real_output + 1e-10))
        fake_loss = tf.reduce_mean(tf.math.log(1 - fake_output + 1e-10))
        return -(real_loss + fake_loss)

    def train(self, data, epochs=200, batch_size=64, pretrain_epochs=100):
        # Pretraining autoencoder
        print("Pretraining autoencoder...")
        for epoch in range(pretrain_epochs):
            for i in range(0, len(data), batch_size):
                batch_data = data[i:i + batch_size]
                with tf.GradientTape() as tape:
                    encoded = self.encoder(batch_data, training=True)
                    decoded = self.decoder(encoded, training=True)
                    loss = tf.reduce_mean(tf.square(batch_data - decoded))
                
                gradients = tape.gradient(
                    loss,
                    self.encoder.trainable_variables + self.decoder.trainable_variables
                )
                self.autoencoder_optimizer.apply_gradients(
                    zip(gradients,
                        self.encoder.trainable_variables + self.decoder.trainable_variables)
                )
            
            if (epoch + 1) % 10 == 0:
                print(f"Pretrain epoch {epoch + 1}, Loss: {loss:.4f}")
        
        # Main training loop
        print("\nTraining GAN...")
        for epoch in range(epochs):
            gen_losses = []
            disc_losses = []
            
            for i in range(0, len(data), batch_size):
                batch_data = data[i:i + batch_size]
                g_loss, d_loss = self.train_step(batch_data, len(batch_data))
                gen_losses.append(g_loss)
                disc_losses.append(d_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}, Gen Loss: {np.mean(gen_losses):.4f}, "
                      f"Disc Loss: {np.mean(disc_losses):.4f}")

    def generate(self, n_samples):
        noise = tf.random.normal([n_samples, self.noise_dim])
        generated_embeddings = self.generator(noise, training=False)
        generated_data = self.decoder(generated_embeddings, training=False)
        return generated_data.numpy()