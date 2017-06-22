from sklearn.cluster import KMeans
from sklearn.externals import joblib
import cv2
import numpy as np
import sys
import glob

# Given a list of images finds and returns all keypoints and descriptors as a vector
def calcVectors(img_list):
    curr_n = 1
    max_n = len(img_list)

    print "Image", curr_n, "of", max_n

    # calculates the kps of the first image before the loop
    curr_img = cv2.imread(img_list[0], 0)
    kp_vector, des_vector = sift.detectAndCompute(curr_img, None)
    for i in img_list[1:]:
        curr_n += 1

        print "Image", curr_n, "of", max_n

        curr_img = cv2.imread(i, 0) # loads image from disk
        kp_i, des_i = sift.detectAndCompute(curr_img, None) # detect its keypoints

        kp_vector = np.append(kp_vector, kp_i, axis=0)
        des_vector = np.append(des_vector, des_i, axis=0)

    return (kp_vector, des_vector)

if len(sys.argv) != 4:
    print "Usage", sys.argv[0], "<path-to-root-image-directory> <number-of-words> <path-to-write-file>"
    sys.exit(1)

# retrieving arguments from command line
root_path = sys.argv[1]
N_CLUSTERS = int(sys.argv[2])
save_path = sys.argv[3]

category = glob.glob(root_path + "*") # retrieves directories inside path which are the categories
category = [i.split("/")[-1] for i in category] # eliminates absolute path

#image_names = glob.glob(sys.argv[1] + "*.png") + glob.glob(sys.argv[1] + "*.jpg") # this is the list of images that will be used to generate the Bag of Features file

sift = cv2.xfeatures2d.SIFT_create()

# dictionary that holds each category and another dictionary for its keypoints and descriptors.
# e.g. "'category': {'kp': kp_list, 'des': des_list}"
cats_dict = {c: dict.fromkeys(['kp', 'des']) for c in category} 

# Now we pass through each image in each category and finds its keypoints
# storing them in the cats_dict. Using the same loop, we also find the bottleneck
# category, i.e. the category with the lowest number of keypoints
print "Calculating keypoints for passed categories..."

bneck_cat = None
bneck_value = float("inf")
for c in category:
    print "Calculating features for category", c
    image_names = glob.glob(root_path + c + "/*.jpg") + glob.glob(root_path + c + "/.png")

    kp_vector, des_vector = calcVectors(image_names)
    
    # defining category with the lowest number of keypoints
    if len(kp_vector) < bneck_value:
        bneck_value = len(kp_vector)
        bneck_cat = c

    cats_dict[c]['kp'] = kp_vector
    cats_dict[c]['des'] = des_vector

print "#####################"
for c in cats_dict:
    print "Category", c, "/ Number of features:", len(cats_dict[c]['kp'])
print "#####################"

print "Category with lowest number of features is", bneck_cat, "with", bneck_value, "features."
n_descriptors = int(0.8*bneck_value)

# Now we need to retrieve the n_descriptors strongest descriptors of each category
# and cluster them using k-means, to build our Bag of Features

# first, we sort the kp list using the response value for each category
for c in category:
    kp_list = cats_dict[c]['kp'] = sorted(cats_dict[c]['kp'],
                                          key=lambda x: x.response,
                                          reverse=True)
    #kp_list = sorted(kp_list, key=lambda x: x.response, reverse=True)
    #cats_dict[c]['kp'] = kp_list

print "Retrieving", n_descriptors, "descriptors (80% of bottleneck value) from each category to build Bag of Features."
# now we mount a vector with the strongest descriptors from each category
# and cluster them using k-means
des_vector = cats_dict[category[0]]['des'][0:n_descriptors]
for c in category[1:]:
    curr_des = cats_dict[c]['des'][0:n_descriptors]
    des_vector = np.append(des_vector, curr_des, axis=0)

print "\n####################"
print "Building Bag of Features..."
print "Clustering", len(des_vector), "features to", N_CLUSTERS, "visual words."

des_vector = np.float64(des_vector) # cast to float64 because kmeans is stupid
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0).fit(des_vector)

print "Done. Dumping Bag of Features to", save_path
joblib.dump(kmeans, save_path)
