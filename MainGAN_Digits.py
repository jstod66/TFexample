import tensorflow as tf
from tensorflow.keras import layers
import glob
import imageio
import os
import PIL
import time
import numpy as np 
import matplotlib.pyplot as plt 
from IPython import display

def make_generator_model():

	model = tf.keras.Sequential()
	#Noise comes into a dense layer of nodes
	model.add(layers.Dense(7*7*256, use_bias = False, input_shape = (100,)))
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())
	
	#Reshape noise into 3-dimensional space
	model.add(layers.Reshape((7, 7, 256)))
	assert model.output_shape == (None, 7, 7, 256) #None is batch size

	#First convolutional layer 
	model.add(layers.Conv2DTranspose(128, (5,5), strides = (1,1), padding = 'same', use_bias = False))
	assert model.output_shape == (None, 7, 7, 128)
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())

	#Second convolutional layer
	model.add(layers.Conv2DTranspose(64, (5,5), strides = (2,2), padding = 'same', use_bias = False))
	assert model.output_shape == (None, 14, 14, 64)
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())

	#Output layer of generator, activation is to force everything between -1 and 1 to make a valid image
	model.add(layers.Conv2DTranspose(1, (5,5), strides = (2,2), padding = 'same', use_bias = False, activation = 'tanh'))
	assert model.output_shape == (None, 28, 28, 1)

	return model

def make_discriminator_model():

	model = tf.keras.Sequential()

	model.add(layers.Conv2D(64, (5,5), strides=(2,2), padding='same',input_shape=[28,28,1]))
	model.add(layers.LeakyReLU())
	model.add(layers.Dropout(0.3))

	model.add(layers.Conv2D(128, (5,5), strides=(2,2), padding='same'))
	model.add(layers.LeakyReLU())
	model.add(layers.Dropout(0.3))

	model.add(layers.Flatten())
	model.add(layers.Dense(1))

	return model

#Discriminator loss compares the real output predictions to an array of ones, and the fake output predictions to an array of zeros
def discriminator_loss(real_output, fake_output):
	real_loss = cross_entropy(tf.ones_like(real_output),real_output)
	fake_loss =	cross_entropy(tf.zeros_like(fake_output),fake_output)
	total_loss = real_loss + fake_loss
	return total_loss

#Generator loss just returns cross entropy of discriminator outputs on fake images compared to array of 1s
def generator_loss(fake_output):
	return cross_entropy(tf.ones_like(fake_output),fake_output)

@tf.function
def train_step(images):

	#Generate as many noise inputs as the batchsize, so the number of generated images is 
	#the same as the number of true images in any given training step
    noise = tf.random.normal([BATCHSIZE, noise_dim])

    #Use gradienttape to keep track of gradients in the evaluation of gen_loss and disc_loss
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    	#Note training is set to 'True' here so that dropout regularization is applied
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    #Extract the gradients from gradienttape record
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    #Apply the gradients to each trainable variable using the chosen optimizer
    generator_opt.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_opt.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss

def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    print('training epoch %d' % (epoch+1))

    gen_loss_ave = 0.0
    count = 0.0
    for image_batch in dataset:
    	gen_loss_ave += train_step(image_batch)
    	count += 1
    gen_loss_ave = gen_loss_ave/count

    # Produce images for the GIF as we go
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    # Save the model every 5 epochs
    if (epoch + 1) % 10 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec, generator loss was {}'.format(epoch + 1, time.time()-start,gen_loss_ave))

  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs,
                           seed)

def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode with correct weight scalings.
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.close()

def display_image(epoch_no):
  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

#Indicate whether training should start from a checkpoint or from a new random initialization
FreshStart = False

#Import the training images from MNIST dataset, reshape and normalize
(train_images, train_labels), (_,_) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0],28,28,1).astype('float32')
train_images = (train_images - 127.5)/127.5

#Batch and shuffle the data??
BUFFERSIZE = 60000
BATCHSIZE = 256
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFERSIZE).batch(BATCHSIZE)

generator = make_generator_model()
discriminator = make_discriminator_model()

#Create a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

#Specify optimizers 
generator_opt = tf.keras.optimizers.Adam(1e-4)
discriminator_opt = tf.keras.optimizers.Adam(1e-4)

#Set up of checkpoints to save and restore training progress
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(generator_opt = generator_opt,
	discriminator_opt = discriminator_opt, 
	generator = generator,
	discriminator = discriminator)

if(not FreshStart):
	print('Loading from latest checkpoint...')
	checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

#Create an input noise vector to test the generator
noise = tf.random.normal([1, 100])
generated_image = generator(noise, training = False)
decision = discriminator(generated_image, training = False)

plt.imshow(generated_image[0, :, :, 0], cmap = 'gray')
plt.title('Generator image before training \n Decision = %f' % decision)
plt.show()

#Set up some experiment variables
EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16
#matrix of seeds will be reused in each iteration to better visualise training progress
seed = tf.random.normal([num_examples_to_generate, noise_dim])

train(train_dataset, EPOCHS)

display_image(EPOCHS)











