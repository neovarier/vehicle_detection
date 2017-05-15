##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/color_space_explore.jpg
[image3]: ./examples/spatial_bin.jpg
[image4]: ./examples/color histogram.jpg
[image5]: ./examples/HOG.png
[image6]: ./examples/sliding_window.png
[image7]: ./examples/heat_bbox.png
[image8]: ./examples/false_positives.png
[image9]: ./examples/label.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

I have referred the code provided in the lessons.

###Feature experimentation: Histogram of Oriented Gradients (HOG) & Color Histogram
I read the vehicle and non-vehicle images.
I explored the data set to get the following information:
* No. of car images
* No. of not-car images
* Image resolution image

Randomly selected 5 of each class for experimentation.

I tried various color spaces to see which one differentiates the colors of the car from the non-cars.
The color space of YCrCb turned out to fairly separate out space for the car colors.
The following images show the color space graph:

I checked the color histogram of car and non-car images with YCrCb color space.
The histogram plots augment the feature set to classify the car and non-car images
I am using np.histogram() function to store the color histogram features in color_hist() function in cell no.
I am X bins for histogram of each channel of YCrCB converted images.
The following are the examples histogram plots:


Also checked with spatial binning feature.
As explained in the lesson, resized images to (32x32) is still recognizable for the human eye.
Using cv2.resize().ravel(), I am extracting the spatial binning feature in bin_spatial() in cell no.
The relavent features are still retained in this feature which could be used for classification.


HOG feature helps in giving a signature for the shape of objects which are color and scale invariant.
As explained in the lessons, it definitely helps in classfying the shapes of car.
I used the scimage.hop() function to extract hog features.
The hog features are getting extracted in the function get_hog_features() in cell no.
This is is used separately for each channel if YCbCr converted image.
The following images show the HOG features of a few car and non-car images.


I am using the following parameters for extracting hog features:
orient = 
pix_per_cell = 
cell_per_block = 


###Classification method

For classification the following steps are executed:
* Extract features from dataset
* Normalize the feature set
* Split the dataset for training and testing and randomly shuffling
* Training the classifier and find the test accuracy with test set

Using the following extracted feature, I concatenate them into a single array using np.hstack():
* Spatial binning
* Color Histogram
* HOG feature

Since each feature would be having different scales, I normalized them using StandardScaler library of sklearn.
The normalization is covered in find_cars() for inference and extract_features() for training.
This feature normalization is carried to avoid the biasing of any particular feature.
The following are the number of the dataset:

I am using the train_test_split() from sklearn.cross_validation library to split the data into training and testing dataset
And also shuffling the dataset before training.
I am using 80:20 ratio train:test split-up.

For classifier I am using Linear SVM from sklearn.svm library.
The training is carried in cell no.
The testing accuracy with the linear svm, I was able to achieve 99.2 % accuracy.

The trained model & parameters for feature extraction are stored in pickle file.
Same pickle is used at the time of car detection in the project video

###Sliding Window Search

The sliding window approach is used in cell no. in function find_cars().
Following are the specifications of the sliding window:
size: 64x64
overlap: 1 cell = 8 pixels in both x and y direction
region of interest: y_start = 380; y_end = 656

As the training images are of 64x64 resolution, I am using a sliding window of 64x64 resolution.
By using small overlap, there is more chance of classifying of cars in the image.
It also helps in detecting multiple windows for the same car which boosts the confidence of detected cars
and helps in rejecting false positives.

Following algorithm is executed ind find_cars() in cell no. :
* Set window at the starting position in the region of interest
* Extract the features (HOG, color histogram & spatial bin) for this window. Normalize the features and concatenate them.
* Use the trained classifier to predict whether the window contains a car or not.
* If the window contains a car, then store the window bounding box
* Slide the window to the next position as per the step size.
* Follow this till the window reaches the end corner of region of interest.

Extracting HOG features could be an expensive operation if used separately for each window.
To accelerate this feature extraction step, I am employing the HOG Sub-Sampling Window Search.
In this method, the HOG features are calculated for the entire image just once for each channel.
hog() from sklearn library is used with feature_vec=False. 
This will return the hog feature image with the same resolution as that of the image instead of returning a flattened array.
In this hog feature image, the hog features are extracted for all the cells present in a particular window.
This will give the hog features for that particular window.
Similarly hog features can be extracted for each window by sliding it in the region of intereset of the hog feature image.

Training with more dataset, improved the accuracy of the classifier. This helped in detecting more windows for the same car
and reduced the false positive detection.

Following is an example image of the detected windows in the test image.

###Heat Map
As described in the lesson, heat map is generated for overlapping windows
The heat map is calculated in the following manner:

* Initialized a heat map image which is the same resolution as the project image
* Traversed through the bounding box. With each pixel inside a bounding box, increment the pixel value by 1 in add_head().
* The region where many windows are overlapping would end up having high pixel values.
* Applied a lower threshold the resultant heat map image, and allow only pixels values only greated than the threshold in apply_threshold()
* Used label function from scipy.ndimage.measurements library to label unique hot regions in the heat map.
* Based on labeled hot regions a tight bounding box is computed in draw_labeled_box()
* Rendered the bounding box using the cv2.rectangle()

The following images show the final bounding boxes detected for the test images

### Video Implementation
Please find the uploaded output_video.mp4 which is detecting the cars in the video.
The pipelines in executed in process_image() in cell no. .

###Improvement Methods used
With the above steps I was facing the following problems:
* The bounding boxes in the output video where very jettery and wobbling accross frames.
* There were a few false positives.

To smooth out the bounding boxes, I used bounding boxes computed from the sliding window approach for the past 10 images and applied heat map approach for the current frame.
This smoothened the bounding boxes accross frames.
This also eliminated the false positives
