# Description: Calculates the feature histogram of a single given image from
# an also given Bag file.

from sklearn.externals import joblib
from sklearn.cluster import KMeans
import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt

if len(sys.argv) != 3:
    print "Usage:", sys.argv[0], "<bag-of-features> <image>"
    sys.exit(1)

query_image = sys.argv[2]
bag = joblib.load(sys.argv[1])

# showing points and labels in the first image
original = cv2.imread(query_image)
qImage = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute(qImage, None)

hist = dict.fromkeys(bag.labels_, 0)
print "Calculating histogram for queried image."
for i in range(0, len(kp)): # len(kp) == len(des)
    x, y = (int(kp[i].pt[0]), int(kp[i].pt[1]))
    label = bag.predict(des[i].reshape(1, -1))
    hist[label[0]] += 1

print "Queried image hist from visual words:"
print hist

plt.subplot(211)
plt.bar(hist.keys(), hist.values(), color='g')

plt.subplot(212)
original = cv2.cvtColor(original, cv2.COLOR_RGB2BGR) # convert to rgb for pyplot
plt.imshow(original)

plt.show()
