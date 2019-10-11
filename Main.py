import tensorflow as tf
from tensorflow import keras

import numpy as np 
import matplotlib.pyplot as plt 

#print(tf.__version__)

def main():

	#Import a database of images for classification example
	fashion_mnist = keras.datasets.fashion_mnist
	(train_images, train_labels), (test_images,test_labels) = fashion_mnist.load_data()

	class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

	#plot some of the training images to show the format
	ExamplePlot(train_images)

	#Scale the images to be between 0 and 1
	train_images = train_images/255.0
	test_images = test_images/255.0

def ExamplePlot(im):
	NUM_TILES = 20
	PIX_WIDTH = 28
	subset = im[:NUM_TILES**2,:,:]
	composite = np.zeros((NUM_TILES*PIX_WIDTH,NUM_TILES*PIX_WIDTH))
	for i in range(0,NUM_TILES):
		for j in range(0,NUM_TILES):
			composite[i*PIX_WIDTH:(i+1)*PIX_WIDTH,j*PIX_WIDTH:(j+1)*PIX_WIDTH] = subset[i+j*NUM_TILES,:,:]
	plt.figure()
	plt.imshow(composite)
	plt.colorbar()
	plt.grid(False)
	plt.show()

main()