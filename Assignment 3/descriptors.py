import cv2 as cv
import os

class HOG:
    def detectAndCompute(self, image, dummy=1):
        ## This function takes an RGB image as input. First, the image is turned into Gray scale. Then, HOG is calculated.
        resize=(128,128)
        grid_size=(8,8)
        num_of_bins=60

        grid_dim = int(resize[0]/grid_size[0])

        # Resize the image to a fixed size:
        image = cv.resize(image,resize)

        # Convert the BGR to grayscale format:
        image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)

        # Create an empty array for the descriptor with proper dimensions:
        hist = np.zeros((grid_size[0]*grid_size[1] , num_of_bins))

        # Calculate the gradients in x and y axis:
        Gradx = cv.Sobel(image,cv.CV_32F,1,0,ksize=3) # By using cv2.CV_32F, gradient of each pixel will be 32-bit floating numbers
        Grady = cv.Sobel(image,cv.CV_32F,0,1,ksize=3)

        # Find the angles between the gradients in radians:
        GradRadian = np.arctan2(Grady,Gradx) # Each element is between -3.14 and 3.14

        # Create an array that contains centers of grids (dummy keypoints):
        dummy_kps = np.zeros((grid_size[0]*grid_size[1],2))

        # Scan the image by the windows and find the histogram of each window. Then, concatenate them column wise:
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):

                dummy_kps[i*grid_size[0]+j,0] = i*grid_dim + grid_dim/2
                dummy_kps[i*grid_size[0]+j,1] = j*grid_dim + grid_dim/2

                # Create a mask that has the same size with images. The mask will be dark (0) in all areas except the
                # desired window location. Window will be white (255).
                mask = np.zeros(resize, np.uint8)
                mask[int(i*(resize[0]/grid_size[0])) : int((i+1)*(resize[0]/grid_size[0])), int(j*(resize[1]/grid_size[1])) : int((j+1)*(resize[1]/grid_size[1]))] = 255

                # Find the L1 norm histograms of each window:
                hist_window = cv.calcHist([GradRadian],[0],mask,[num_of_bins],[-np.pi,np.pi])
                hist_window = hist_window / np.sum(np.abs(hist_window)) # Divide the histogram array by its L1 norm, so it adds up to 1.

                # Add the histogram of current cell to the overall descriptor matrix:
                hist[i*grid_size[0]+j] = hist_window[:,0]

        return dummy_kps, hist


def get_descriptor(descname):
    # https://docs.opencv.org/3.4/d0/d13/classcv_1_1Feature2D.html
    if descname == 'SIFT':
        return cv.xfeatures2d.SIFT_create()
    elif descname == 'SURF':
        return cv.xfeatures2d.SURF_create()
    elif descname == 'ORB':
        return cv.ORB_create()
    elif descname == 'HOG':
        return HOG()
    else:
        raise Exception('Invalid option')

def features_in_dir(descriptor, directory, print_progress=False):
    features  = []
    imnames = os.listdir(directory)
    for i, imname in enumerate(imnames, start=1):
        if print_progress:
            print('[%d/%d] Reading %s' % (i, len(imnames), imname))
        impath = os.path.join(directory, imname)
        img = cv.imread(impath)
        assert img is not None, 'Could not open image %s' % (impath)
        feature = descriptor.detectAndCompute(img, None)
        features.append(feature)
    return features

if __name__ == '__main__':
    import imgops
    # parameters
    descname = 'SIFT'
    imname = 'Dataset/Caltech20/training/airplanes/image_0001.jpg'
    # read an image
    img = cv.imread(imname)
    assert img is not None
    # choose a descriptor and find keypoints
    descriptor = get_descriptor(descname)
    kps, desc = descriptor.detectAndCompute(img, None)
    # display the results
    print('descriptors:', desc.shape)
    print(desc)
    disp_img = cv.drawKeypoints(img, kps, None)
    imgops.plt_imshow(disp_img)
