# Import TensorFlow and TensorFlow Datasets
import tensorflow as tf
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

# Helper libraries
import math
import numpy as np
import matplotlib.pyplot as plt
#print(tf.__version__)

################################################################################# Auxiliary Functions

def normalize(images, labels):
  images = tf.cast(images, tf.float32)
  images /= 255
  return images, labels

def plot_image(i, predictions_array, true_labels, images):
  predictions_array, true_label, img = predictions_array[i], true_labels[i], images[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img[...,0], cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)

  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
    print("predicted_label is ",predicted_label,"(", class_names[predicted_label],") and the true_label is ",true_label,"(", class_names[true_label],")")


  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

############################################################################################### Load the Fashion MNIST dataset
dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples
print("Number of training examples: {}".format(num_train_examples))
print("Number of test examples:     {}".format(num_test_examples))

# The map function applies the normalize function to each element in the train
# and test datasets
train_dataset =  train_dataset.map(normalize)
test_dataset  =  test_dataset.map(normalize)


################################################################################ Exploring the data
# from numpy.ma.core import shape
# # Take a single image, and remove the color dimension by reshaping
# for image, label in test_dataset.take(1):
#   break
# aa = image
# image = image.numpy().reshape((28,28))
#
# print("image after :",shape(image),"image before :",shape(aa))
#
# # Plot the image - voila a piece of fashion clothing
# plt.figure()
# plt.imshow(image ,cmap=plt.cm.binary)
# plt.colorbar()
# plt.grid(False)
# plt.show()

##############################################

# plt.figure(figsize=(10,10))
# i = 0
# for (image, label) in test_dataset.take(25):
#     image = image.numpy().reshape((28,28))
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(image, cmap=plt.cm.binary)
#     plt.xlabel(class_names[label])
#     i += 1
# plt.show()
################################################################################ Build and Train the model

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10,  activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

BATCH_SIZE = 32
train_dataset = train_dataset.repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

model.fit(train_dataset, epochs=5, steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE))

################################################################################# Evaluate accuracy

test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test_examples/BATCH_SIZE))
print('Accuracy on test dataset:', test_accuracy)

################################################################################# Predictions (Single Image)

for test_images, test_labels in test_dataset.take(1):
  test_images = test_images.numpy()
  test_labels = test_labels.numpy()
  predictions = model.predict(test_images)

#print(predictions.shape) #(32, 10) 32 arrays (as the number of pictures) of 10 possible answers
#print(predictions[0]) #[..,..,..,..,..] array of 10 probabilities of the 0th image
#print(np.argmax(predictions[0])) #(4) the position of the biggest number in the array
#print(test_labels[0]) #(4) the first label in the test_dataset

img = test_images[1]
print("shape before matching", img.shape)
plt.figure()
plt.imshow(img ,cmap=plt.cm.binary)
plt.show()
# tf.keras models are optimized to make predictions on a batch, or collection,
# of examples at once. So even though we're using a single image, we need to add it to a list
img = np.array([img])
print("shape after matching",img.shape)
######################

predictions_single = model.predict(img)

#print(shape(predictions_single)) #(1,10) ONE array of 10 probabilities
#print(type(predictions_single)) #<class 'numpy.ndarray'>
#print(predictions_single) #[[..,..,..,..,..,..,..]] array inside an array of 10 probabilities
#print(predictions_single[0]) #[..,..,..,..,..,..,..] array inside an array of 10 probabilities
#print(predictions_single[0][4]) #one number from 10 probabilities

plt.figure()
plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=90)
plt.show()

################################################################################## few examples (Predictions Multiple Images)

# i = 0
# plt.figure(figsize=(6,3))
# plt.subplot(1,2,1)
# plot_image(i, predictions, test_labels, test_images)
# plt.subplot(1,2,2)
# plot_value_array(i, predictions, test_labels)

#################################################

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()

#######################################################################################