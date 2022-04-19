import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn import svm
import sys
from glob import glob
from matplotlib import pyplot as plt

if len(sys.argv) != 4:
    print("Usage:", sys.argv[0], "<path-to-root-image-directory> <bag-of-features> <classifier-save-file>")
    sys.exit(1)

root_path = sys.argv[1]
bag = joblib.load(sys.argv[2])
save_file = sys.argv[3]

vocab_size = len(set(bag.labels_)) # how many words there are in the bag

category = glob(root_path + "*") # retrieves directories inside root path
category = [i.split("/")[-1] for i in category] # eliminates absolute path

sift = cv2.SIFT_create()

# First we need to mount the histogram vector containing the images from all
# categories 
# Counting number of images
num_images = 0
for c in category:
    num_images += len(glob(root_path + c + "/*")) 

print("Generating feature histograms for", num_images, "images.")

hist_vector = np.zeros((num_images, vocab_size)) # will hold all histograms from all images
hist_pos = 0 # var that points to next position to be filled in hist_vector
targets = []
# Now, we iterate through the images in the categories calculating the histograms
# for each image using the bag of features, filling the histogram vector and target/label
# vector
for c in category:
    print("Category:", c)
    c_imgs = glob(root_path + c + "/*")

    for i in c_imgs:
        print("Image", hist_pos+1, "of", num_images)

        curr_img = cv2.imread(i, 0)
        kp_i, des_i = sift.detectAndCompute(curr_img, None)
        
        # mounts histogram for current image
        hist_i = np.zeros(vocab_size)
        word_vector = bag.predict(np.asarray(des_i, dtype=float)) # uses the bag of words to find which cluster this descriptor best fit

        # for each unique word
        for word in np.unique(word_vector):
            res = list(word_vector).count(word) # count the number of word in word_vector
            hist_i[word] = res # increment the number of occurrences of it

        # normalizes histogram
        cv2.normalize(hist_i, hist_i, norm_type=cv2.NORM_L2)
        #hist_i = hist_i/hist_i.sum()

        hist_vector[hist_pos] = hist_i # puts current histogram into hist_vector
        targets.append(c) # append current category label to the targets
        hist_pos += 1

print("###########################")
print("Finished generating feature histograms!")
print("###########################")
print("Fitting data histograms to labels/Training classifier...")

classifier = svm.SVC(probability=True)

print(hist_vector)
print("###############")

#dataScaler = StandardScaler()
#dataScaler.fit(hist_vector)
#params = dataScaler.get_params()
#hist_vector = dataScaler.transform(hist_vector)

#print hist_vector

print(targets)
classifier.fit(hist_vector, targets)

print("Done! Dumping classifier to file:", save_file)
joblib.dump(classifier, save_file)
#joblib.dump(dataScaler, "scaler")

#plt.figure(1)
#plt.bar(np.arange(len(hist_vector[0])), hist_vector[0], color='r')
##
#plt.show()
