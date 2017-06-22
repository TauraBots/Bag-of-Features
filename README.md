# Bag-of-Features
Implementation of image classification using Bag of Features model.

The goal for this project is to provide means for identification of objects within predefined categories in real-time using a video camera for Dimitri.
In the Bag of Features model we create a dictionary of visual words, elements which compose an image, and then use it to generate a histogram of each "word" occurence in an image, the histogram being a new, more concise representation of the image. After the dictionary is built we train a classifier based on the histograms of images.
This method is pretty good at classifying an image, however finding the localization of the objects within the image is still an important issue. To explore in this idea we will be implementing a kind of "Convolutional Classifier" which basically is a dinamically sized window which will pass through the image labeling each slices individually, allowing us to find the localization (bounding box) of an object and multiple objects within the same image.

There are many things to keep in mind while developing this model, the most important being response time (time it takes to find and label objects) and accuracy. For now, we are using SIFT to find the keypoints and calculate its descriptors. More approaches using SURF, Orb, etc., will be tried and measured later.

## How to use

### Create the Bag of Features
> python createBag.py path-to-root-image-directory number-of-words path-to-write-file

The first argument is root directory of the dataset that will be used to generate the bag of features. Keep in mind that each image should be inside the folder corresponding to its category, like in the Example Dataset.

The second argument is the number of clusters or words that will be generated from all the keypoints and descriptors in the dataset. Usual values are 500~1000.

The third argument is the filename which to save the bag of features.

### Train the Classifier
> python trainClassifier.py path-to-root-image-directory bag-of-features classifier-save-file

The first argument is root directory of the dataset that will be used to train the classifier. The images should be separated within folders whose names are the categories of images within. These will be the labels for the supervised training and the categories which the objects will be classified later.

The second argument is the bag of features file that was created earlier.

The third argument is filename which to save the classifier.

### Classify an Image
> python classifyImage.py classifier-file bag-of-features-file query-image

The first argument is the trained classifier.

The second argument is the bag of features. It's important that the classifier and bag match, that means, the bag passed should be the same that was used when training the classifier.

The third argument is the image to be classified.

## Implementation

Meh, I'll write this later, basically KMeans for the clustering/generating bag of features, and an SVM for the classifier.

## TODO
* Write the convolutional classifier.
* Other things.
