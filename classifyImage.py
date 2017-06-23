# Description: loads an image, a bag of features and a svm classifier from disk, find and compute
# keypoints and descriptors for the given image, generate the feature histogram
# and feeds it to the classifier

import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.cluster import KMeans
from sklearn import svm
import sys
from matplotlib import pyplot as plt

if len(sys.argv) != 4:
    print "Usage", sys.argv[0], "<classifier-file> <bag-of-features-file> <query-image>"
    sys.exit(1)

classifier = joblib.load(sys.argv[1])
bag = joblib.load(sys.argv[2])
query_image = cv2.imread(sys.argv[3])

if query_image.shape[2]: # rgb images should be converted to grayscale for sift
    query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute(query_image, None)

vocab_size = len(set(bag.labels_)) # how many visual words there are in the bag
hist = np.zeros(vocab_size) # initializes the histogram with zeros
for i in range(0, len(kp)): # iterates through keypoints counting our visual words
    word = bag.predict(des[i].reshape(1, -1)) # finds which of our words fit best
    hist[word[0]] += 1 # increment one occurence of it

# normalizes histogram
cv2.normalize(hist, hist, norm_type=cv2.NORM_L2)

category = classifier.predict([hist])
print "This is:", category
print "proba:", classifier.decision_function([hist])
