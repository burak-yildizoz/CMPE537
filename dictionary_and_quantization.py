import numpy as np
from scipy.cluster.vq import vq
from numpy.linalg import norm

def findDictionary(data, k=100, max_iter=10):

    # random initialization
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]

    ind = np.zeros((data.shape[0], 1))

    for iter_num in range(max_iter):

        for i in range(data.shape[0]):
            delta = norm((data[i, :] - centroids), axis=1)
            ind[i, :] = np.argmin(delta)

        for j in range(k):
            centroids[j, :] = np.mean(data[ind[:, 0] == j, :], axis=0)
    return centroids

def featureQuantization(descriptor_list, centroids):
    num_images = len(descriptor_list)
    num_words = centroids.shape[0]
    
    im_features = np.zeros((num_images, num_words), "float32")
    for i in range(num_images):
        words, _ = vq(descriptor_list[i], centroids)
        for w in words:
            im_features[i][w] += 1
            
    # Normalize the image features such that L1 norm of each image is 1:
    im_features = im_features / np.sum(np.abs(im_features) , axis=1).reshape((im_features.shape[0],1))
        
    return im_features