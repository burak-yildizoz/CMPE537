import numpy as np
import data_read
import dictionary_and_quantization
import classification2
import evaluation
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt


# Parameters to be chosen:
descriptor = 'ORB'
k = 100 # number of words in the dictionary
max_iter = 5 # number of iterations for the k-means algorithm applied for finding the dictionary
classifier_type = 'MLP'
#parameters = [400] # regularization parameter for SVM
parameters = [(100, )] # hidden layer info for  MLP
#parameters = [5] # k parameter for kNN

# training_directory = 'C:/Users/oguzh/Desktop/Graduate_Courses/CMPE 537 Computer Vision/HW3/Caltech20/training'
# test_directory = 'C:/Users/oguzh/Desktop/Graduate_Courses/CMPE 537 Computer Vision/HW3/Caltech20/testing'

training_directory = 'C:/Users/Umit/PycharmProjects/CmpE537_HW3/Caltech20/training'
test_directory = 'C:/Users/Umit/PycharmProjects/CmpE537_HW3/Caltech20/testing'

# Read the training and test data. While reading each image, the descriptors are extracted as well.
print('Reading the training data may take a while. Please wait...')
training_descriptors, X, y, classNameDic = data_read.DataReadTrain(training_directory, descriptor=descriptor)
print('Training data is read')
X_test, y_test = data_read.DataReadTest(test_directory, classNameDic=classNameDic, descriptor=descriptor)
print('Test data is read')

# Find the dictionary by using the training descriptors:
centroids = dictionary_and_quantization.findDictionary(training_descriptors, k=k, max_iter=max_iter)
print('Dictionary is learnt')

# Quantize the descriptors of training and test sets by using the centroids found in previous step:
X_train = dictionary_and_quantization.featureQuantization(X, centroids)
X_test = dictionary_and_quantization.featureQuantization(X_test, centroids)
print('Quantization is done')

# Train the classifier:
model = classification2.TrainClassifier(classifier_type, parameters, X=X_train, y=y)
y_predict = classification2.TestClassifier(classifier_type, model, X_test)
print('CLass predictions of test data is done')

# Calculate evaluation scores:
MeanF1, precision, recall, F1, conf_matrix = evaluation.Scores(y_test, y_predict)
accuracy = 100 * np.sum(y_predict == y_test) / y_test.shape[0]

print("MeanF1:", MeanF1)
print("precision:", precision)
print("recall:",recall)
print("F1:", F1)
print("conf_matrix:", conf_matrix)
print("accuracy:", accuracy)

plot_confusion_matrix(model, X_test, y_test)
plt.show()
