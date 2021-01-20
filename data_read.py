import os
import cv2 as cv
import numpy as np

def HOG(image):
    ## This function takes an RGB image as input. It calculates HOG features at each keypoint found by SIFT.
    max_kp = 500
    win_size = 12
    num_of_bins=30
    
    # Width and height of the image:
    im_h = image.shape[0]
    im_w = image.shape[1]
    
    # Find the coordinates of SIFT keypoints:
    sift = cv.SIFT_create()
    kp = sift.detect(image,None)
    kp = np.array([kp[i].pt for i in range(len(kp))])
    
    # IF there exists too many keypoints, select random max_kp of them.
    if kp.shape[0] > max_kp:
        idx = np.random.permutation(kp.shape[0])[0:max_kp]
        kp = kp[idx,:]
        
    # Convert the BGR to grayscale format:
    image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    
    # Calculate the gradients in x and y axis:
    Gradx = cv.Sobel(image,cv.CV_32F,1,0,ksize=3) # By using cv2.CV_32F, gradient of each pixel will be 32-bit floating numbers
    Grady = cv.Sobel(image,cv.CV_32F,0,1,ksize=3)
    
    # Find the angles between the gradients in radians:
    GradRadian = np.arctan2(Grady,Gradx) # Each element is between -3.14 and 3.14
    
    # Create an empty array for the descriptor with proper dimensions:
    hist = np.zeros((kp.shape[0] , num_of_bins)) 
    
    for i in range(kp.shape[0]):
        kp_coor = np.rint(kp[i,:])
        
        # Create a mask around the keypoint i:
        mask = np.zeros(image.shape, np.uint8)
        mask[int( np.max((kp_coor[1]-win_size,0)) ) : int( np.min((kp_coor[1]+win_size+1,im_h)) ) ,
             int( np.max((kp_coor[0]-win_size,0)) ) : int( np.min((kp_coor[0]+win_size+1,im_w)) ) ] = 1
        
        # Find the histogram of gradients for the window around keypoint i:
        hist_window = cv.calcHist([GradRadian],[0],mask,[num_of_bins],[-np.pi,np.pi])
        hist_window = hist_window / np.sum(np.abs(hist_window)) # Divide the histogram array by its L1 norm, so it adds up to 1.

        # Add the histogram of current cell to the overall descriptor matrix:
        hist[i,:] = hist_window[:,0]

    return kp, hist


def DataReadTrain(direct, descriptor='HOG'):
    y = [] # We will collect the labels of dataset in this list
    X = [] # This will be our dataset
    descs = []
    
    # Keep class names in and corresponding label in this dic.
    classNameDic = {}
    
    for cls, class_name in enumerate(os.listdir(direct)):
        classNameDic.update({class_name: cls})

        print("Class Name:", class_name)
        class_dir = direct + '/' + class_name
        for im_name in os.listdir(class_dir):
            
            # Read the image as a (x,y,3) numpy array:
            image = cv.imread(class_dir + '/' + im_name)
            
            # Find the descriptors of the image:
            if descriptor == 'HOG':
                _, descriptors = HOG(image)
            elif descriptor == 'SIFT':
                _, descriptors = cv.xfeatures2d.SIFT_create().detectAndCompute(image, None)
            elif descriptor == 'ORB':
                _, descriptors = cv.ORB_create().detectAndCompute(image, None)
            if descriptors is not None:
                y.append(cls)
                X.append(list(descriptors))
                descs = descs + list(descriptors)
             
    descs = np.array(descs)
    y = np.array(y)
    return descs, X, y, classNameDic


def DataReadTest(direct, classNameDic, descriptor='HOG'):
    y = [] # We will collect the labels of dataset in this list
    X = [] # This will be our dataset
    
    for class_name in os.listdir(direct):
        cls = classNameDic[class_name]
        class_dir = direct + '/' + class_name
        for im_name in os.listdir(class_dir):
       
            # Read the image as a (x,y,3) numpy array:
            image = cv.imread(class_dir + '/' + im_name)
            
            # Find the descriptors of the image:
            if descriptor == 'HOG':
                _, descriptors = HOG(image)
            elif descriptor == 'SIFT':
                _, descriptors = cv.xfeatures2d.SIFT_create().detectAndCompute(image, None)
            elif descriptor == 'ORB':
                _, descriptors = cv.ORB_create().detectAndCompute(image, None)
            if descriptors is not None:
                y.append(cls)
                X.append(list(descriptors))
                
    y = np.array(y)
    return X, y