def train(D, G, D_solver, G_solver, discriminator_loss, generator_loss,\
              show_every=20, print_every=20, batch_size=128, num_epochs=10, noise_size=100):
    """
    Train loop for GAN.

    The loop will consist of two steps: a discriminator step and a generator step.

    (1) In the discriminator step, first step is to sample noise to generate a fake data batch using the generator. 
    Calculate the discriminator output for real and fake data, and use the output to compute discriminator loss. 
    In code 'd_total_error' is the total discriminator loss.
    
    (2) For the generator step, first step is to sample noise to generate a fake data batch. 
    Get the discriminator output for the fake data batch and use this to compute the generator loss. 
    In code 'g_error' is the generator loss. 
    
    Use the sample_noise function to sample random noise, and the discriminator_loss
    and generator_loss functions for their respective loss computations

    Inputs:
    - D: Discriminator model
    - G: Generator model
    - D_solver: an Optimizer for Discriminator
    - G_solver: an Optimizer for Generator
    - generator_loss: Generator loss
    - discriminator_loss: Discriminator loss
    Returns:
        Nothing
    """
    mnist = MNIST(batch_size=batch_size, shuffle=True)
    
    iter_count = 0
    for epoch in range(num_epochs):
        for (x, _) in mnist:
            with tf.GradientTape() as tape:
                real_data = x
                
                
                z = sample_noise(batch_size, noise_size)
                fake_images = G(z)
                
                real_output = D(preprocess_img(x))
                fake_output = D(fake_images)

                d_total_error = discriminator_loss(real_output, fake_output)

                d_gradients = tape.gradient(d_total_error, D.trainable_variables)      
                D_solver.apply_gradients(zip(d_gradients, D.trainable_variables))
            
            with tf.GradientTape() as tape:
                
                
                z = sample_noise(batch_size, noise_size)
                fake_images = G(z)

                fake_output = D(fake_images)

                g_error = generator_loss(fake_output)

                
                g_gradients = tape.gradient(g_error, G.trainable_variables)      
                G_solver.apply_gradients(zip(g_gradients, G.trainable_variables))

            if (iter_count % show_every == 0):
                print('Epoch: {}, Iter: {}, D: {:.4}, G:{:.4}'.format(epoch, iter_count,d_total_error,g_error))
                imgs_numpy = fake_images.cpu().numpy()
                show_images(imgs_numpy[0:16])
                plt.show()
            iter_count += 1
    
    # random noise fed into our generator
    z = sample_noise(batch_size, noise_size)
    # generated images
    G_sample = G(z)
    print('Final images')
    show_images(G_sample[:16])
    plt.show()
   
if __name__ == "__main__":
    D = discriminator()
    G = generator()

    # Use the function you wrote earlier to get optimizers for the Discriminator and the Generator
    D_solver = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.5, beta_2=0.999)
    G_solver = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.5, beta_2=0.999)

    # Run it!
    train(D, G, D_solver, G_solver, discriminator_loss, generator_loss)