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

if len(sys.argv) != 3:
    print("Usage", sys.argv[0], "<classifier-file> <bag-of-features-file>")
    sys.exit(1)

classifier = joblib.load(sys.argv[1])

bag = joblib.load(sys.argv[2])
vocab_size = len(set(bag.labels_)) # how many visual words there are in the bag

sift = cv2.SIFT_create()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.7, fy=0.7)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kp, des = sift.detectAndCompute(frame, None)

    hist = np.zeros(vocab_size) # initializes the histogram with zeros
    word_vector = bag.predict(np.asarray(des, dtype=float))

    # for each unique word
    for word in np.unique(word_vector):
        res = list(word_vector).count(word) # count the number of word in word_vector
        hist[word] = res # increment the number of occurrences of it

    # normalizes histogram
    cv2.normalize(hist, hist, norm_type=cv2.NORM_L2)

    print(hist)
    print("I see:", classifier.predict([hist]))
    #print "Distance:", classifier.decision_function([hist])

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cv2.destroyAllWindows()
