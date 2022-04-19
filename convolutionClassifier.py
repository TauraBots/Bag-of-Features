# Description: The idea behind this program is using an implementation of a
# bag of features, instead of classifying an image as a whole, we pass through
# the image using a dinamycally-sized "window", classifying each part of the
# image, much like a convolution filter in image processing. The idea is that
# this can classify multiple objects in a single image and information about
# where each objects lies within the image

import cv2
import numpy as np
import sys
from sklearn.externals import joblib
from sklearn.cluster import KMeans
from sklearn import svm
from copy import copy

# given a image returns its representation as an image histogram using
# the bag of features
def getFeaturesHist(img):
    # gets keypoints and its descriptors using SIFT
    kp, des = sift.detectAndCompute(img, None) 

    if len(kp) == 0: # no keypoints detected
        return None

    # mount the features histogram from the descriptors using the bag
    features_hist = np.zeros(vocab_size)
    for d in des:
        word = bag.predict(d.reshape(1, -1))
        features_hist[word[0]] += 1                

    return features_hist

def convolutionClassifier(img):
    height, width = img.shape[:2]

    k_boundaries = (height/8, width/8) # start with a 96x96 kernel size
    curr_x, curr_y = (0, 0)

    # iterates through the image taking slices of the image k_boundaries size
    # and classifying them
    while k_boundaries <= (height, width):

        for curr_x in range(0, width - k_boundaries[1] + 1, k_boundaries[1]):
            for curr_y in range(0, height - k_boundaries[0] + 1, k_boundaries[0]):
                rgb_img = copy(original_img)
                roi = img[curr_y:curr_y + k_boundaries[0],
                          curr_x:curr_x + k_boundaries[1]]
                
                cv2.rectangle(rgb_img, (curr_x, curr_y), 
                              (curr_x + k_boundaries[1], curr_y + k_boundaries[0]),
                              (0, 0, 255), 1)

                roi_hist = getFeaturesHist(roi)
                #
                try:
                    label = classifier.predict([roi_hist])
                    if label[0] == 'bike':
                        print("I see:", classifier.predict([roi_hist]))
                        cv2.imshow("Image", rgb_img)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                    
                except ValueError:
                    pass
                    #print "Meeeh"

                #cv2.imshow("ROI", roi)
                #cv2.imshow("Image", rgb_img)
                #cv2.waitKey(0)

        #curr_x = curr_x + k_boundaries[1]
        #curr_y = curr_y + k_boundaries[0]

        #if curr_x + k_boundaries[1] == width or curr_y + k_boundaries[0] == height:
        #    curr_x = 0
        #    curr_y = 0
        k_boundaries = (k_boundaries[0] * 2, k_boundaries[1] * 2)

if len(sys.argv) != 4:
    print("Usage:", sys.argv[0], "<Bag-of-Features> <Classifier> <Image>")
    sys.exit(1)

# retrieving arguments from command line
bag = joblib.load(sys.argv[1])
classifier = joblib.load(sys.argv[2])
original_img = cv2.imread(sys.argv[3])

# creating sift object
sift = cv2.xfeatures2d.SIFT_create()

vocab_size = len(set(bag.labels_)) # the total number of visual words in the bag

gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

convolutionClassifier(gray)
