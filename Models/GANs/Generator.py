def generator(noise_dim=NOISE_DIM):
    """Generate images from a random noise vector.
    
    Inputs:
    - z: TensorFlow Tensor of random noise with shape [batch_size, noise_dim]
    
    Returns:
    TensorFlow Tensor of generated images, with shape [batch_size, 784].
    """
    model = tf.keras.models.Sequential([
        
        tf.keras.layers.Dense(1024, use_bias=True, input_shape=(NOISE_DIM,), activation='relu'),
        tf.keras.layers.Dense(1024, use_bias=True, activation='relu'), 
        tf.keras.layers.Dense(784, use_bias=True, activation='tanh')


    ])
    return model