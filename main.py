import os
import numpy as np

import descriptors
import dictionary
import quantization
import classification
import evaluation

# parameters
clustername = 'BOW'
num_cluster = 300
descname = 'SIFT'
classifier_name, parameters = 'KNN', [5]
classifier_name, parameters = 'SVM', [1.0]
classifier_name, parameters = 'MLP', [100, 'constant']
traindir = 'Dataset/Caltech20/training'
testdir = traindir + '/../testing'
# collect descriptors
descriptor = descriptors.get_descriptor(descname)
featuresdict = {}
taxons = os.listdir(traindir)
for taxon in taxons:
    taxonpath = os.path.join(traindir, taxon)
    print('Collecting features from', taxon)
    features = descriptors.features_in_dir(descriptor, taxonpath, print_progress=True)
    featuresdict.update({taxon : features})
# obtain vocabulary
descs = np.vstack([desc for _, desc in features for features in featuresdict.values() if desc is not None])
dictionary = dictionary.get_dictionary(clustername, num_cluster)
dictionary.add(descs)
print('Cluster!')
vocab = dictionary.cluster()
# collect histograms
quantizer = quantization.get_quantizer(clustername, descriptor)
quantizer.setVocabulary(vocab)
X, y = [], []
for i, taxon in enumerate(taxons, start=1):
    taxonpath = os.path.join(traindir, taxon)
    print('Calculating histograms of', taxon)
    hists = quantization.hists_in_dir(quantizer, featuresdict[taxon], taxonpath, print_progress=True)
    del featuresdict[taxon]
    for hist in hists:
        X.append(hist)
        y.append(i)
X = np.vstack(X)
# collect histograms of test images
X_test, truth = [], []
tests = os.listdir(testdir)
for taxon in tests:
    try:
        label = taxons.index(taxon) + 1
    except:
        print('train directory does not contain %s but test directory does' % (taxon))
        label = -1
    taxonpath = os.path.join(testdir, taxon)
    print('Collecting features from test directory', taxon)
    features = descriptors.features_in_dir(descriptor, taxonpath, print_progress=True)
    print('Calculating histograms of test directory', taxon)
    hists = quantization.hists_in_dir(quantizer, features, taxonpath, print_progress=True)
    for hist in hists:
        X_test.append(hist)
        truth.append(label)
X_test =  np.vstack(X_test)
# save
np.save('X.npy', X)
np.save('y.npy', y)
np.save('X_test.npy', X_test)
np.save('truth.npy', truth)
# load
X = np.load('X.npy')
y = np.load('y.npy')
X_test = np.load('X_test.npy')
truth =  np.load('truth.npy')
# classify histograms
print('Training model')
model = classification.TrainClassifier(classifier_name, parameters, X, y)
print('Predicting results')
predict = classification.TestClassifier(classifier_name, model, X_test)
print('Classification completed!')
# evaluate results
print('accuracy: %.1f%%' % (100 * np.count_nonzero(predict == truth) / len(truth)))
MeanF1, precision, recall, F1, conf_matrix = evaluation.Scores(truth, predict)
print('MeanF1', MeanF1)
print('precision', precision)
print('recall', recall)
