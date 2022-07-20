# Description: Calculates the feature histogram of a single given image from
# an also given Bag file.

from sklearn.externals import joblib
from sklearn.cluster import KMeans
import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt
from copy import copy

if len(sys.argv) != 3:
    print("Usage:", sys.argv[0], "<bag-of-features> <image>")
    sys.exit(1)

query_image = sys.argv[2]
bag = joblib.load(sys.argv[1])

vocab_size = len(set(bag.labels_))

# showing points and labels in the first image
original = cv2.imread(query_image)
qImage = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create()
kp, des = sift.detectAndCompute(qImage, None)

h = np.zeros(vocab_size)

print("Calculating histogram for queried image.")
word_vector = bag.predict(np.asarray(des, dtype=float))

# for each unique word
for word in np.unique(word_vector):
    res = list(word_vector).count(word) # count the number of word in word_vector
    h[word] = res # increment the number of occurrences of it

# normalizing histogram using OpenCV normalize
normalized = np.zeros(vocab_size)
cv2.normalize(h, normalized, norm_type=cv2.NORM_L2)

print("Queried image hist from visual words:")
    #print hist

plt.subplot(221)
plt.bar(np.arange(len(h)), h, color='g')

plt.subplot(223)
original = cv2.cvtColor(original, cv2.COLOR_RGB2BGR) # convert to rgb for pyplot
plt.imshow(original)

plt.subplot(222)
plt.bar(np.arange(len(normalized)), normalized, color='r')

plt.show()
