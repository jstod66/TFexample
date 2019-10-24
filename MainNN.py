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

	(num_trainimages,trainim_width,trainim_height) = train_images.shape
	(num_testimages,testim_width,testim_height) = test_images.shape

	#plot some of the training images to show the format
	#ExamplePlot(train_images)

	#Scale the images to be between 0 and 1
	train_images = train_images/255.0
	test_images = test_images/255.0

	#Plot the scaled test images with labels, for the first 25 elements
	#ExamplePlotLabels(train_images,train_labels,class_names)

	#Set up the model structure
	model = keras.Sequential([
    keras.layers.Flatten(input_shape=(trainim_width, trainim_height)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
	])

	#Compile the model
	model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

	#Train the model on the training data
	model.fit(train_images, train_labels, epochs=10)

	#Evaluate the accuracy on the validation data
	test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
	print('\nTest accuracy:', test_acc)

	#Form a list of predicted labels
	predictions = model.predict(test_images)
	predictedLabels = np.argmax(predictions,1)
	predictedLabels = [class_names[i] for i in predictedLabels]
	#print(predictedLabels)

	#Prompt for user input to pick a test image and display it along with its predicted class
	prompt_str = "Pick a test image index between 0 and %d: " % num_testimages
	Label_num = input(prompt_str)
	Label_num = int(Label_num)

	plt.figure()
	plt.xticks([])
	plt.yticks([])
	plt.grid(False)
	plt.imshow(test_images[Label_num,:,:], cmap=plt.cm.binary)
	plt.xlabel("Model Prediction: " + predictedLabels[Label_num] + "\n" + "Real Class: " + class_names[test_labels[Label_num]])
	plt.show()

def ExamplePlotLabels(images,labels,class_names):

	plt.figure(figsize=(7,7))
	for i in range(25):
	    plt.subplot(5,5,i+1)
	    plt.xticks([])
	    plt.yticks([])
	    plt.grid(False)
	    plt.imshow(images[i], cmap=plt.cm.binary)
	    plt.xlabel(class_names[labels[i]])
	plt.show()


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