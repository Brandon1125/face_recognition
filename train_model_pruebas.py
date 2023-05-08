
# Loading the dataset
  
from keras.datasets import cifar10
(train_images, train_labels), (test_images, test_labels)= cifar10.load_data()
  
# Labels before applying the function
# Training set labels
print(train_labels)
print(train_labels.shape)
  
# Testing set labels
print(test_labels)
print(test_labels.shape)
  
# Applying the function to training set labels and testing set labels
from keras.utils import to_categorical
train_labels = to_categorical(train_labels, dtype ="uint8")
test_labels = to_categorical(test_labels, dtype ="uint8")
  
# Labels after applying the function
# Training set labels
print(train_labels)
print(train_labels.shape)
  
# Testing set labels
print(test_labels)
print(test_labels.shape)