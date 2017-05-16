
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/1.png
[image2]: ./examples/YCrCb_1.png
[image3]: ./examples/2.png
[image4]: ./examples/YCrCb_2.png
[image5]: ./examples/3.png
[image6]: ./examples/YCrCb_3.png
[image7]: ./examples/car.png
[image8]: ./examples/car_resize.png
[image9]: ./examples/spatial_bin.png
[image10]: ./examples/color_histogram.png
[image11]: ./examples/HOG.png
[image12]: ./examples/HOG_2.png
[image13]: ./examples/HOG_3.png
[image14]: ./examples/window_1.png
[image15]: ./examples/window_2.png
[image16]: ./examples/window_3png.png
[image17]: ./examples/window_4.png
[image18]: ./examples/heatmap_1.png
[image19]: ./examples/heatmap_2.png
[image20]: ./examples/heamap_3.png
[image21]: ./examples/heatmap_4.png
[image22]: ./examples/label_1.png
[image23]: ./examples/label_2.png
[image24]: ./examples/label_3.png
[image25]: ./examples/label_4.png
[image26]: ./examples/test1.png
[image27]: ./examples/test2.png
[image28]: ./examples/test3.png
[image29]: ./examples/test4.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README  

I have referred the code provided in the lessons.
The submission contains the following:

* README.md
* example images
* output_project.mp4
* VehicleDetect.ipynb
* VehicleDetect.html
* train_model.p

###Feature experimentation: Histogram of Oriented Gradients (HOG) & Color Histogram.

I read the vehicle and non-vehicle images.
I explored the data set to get the following information:

* No. of car images:  8792
* No. of non-car images:  8968
* Image resolution:  (64, 64, 3)

I selected few images of each class for experimentation.

I tried various color spaces to see which one differentiates the colors of the car from the non-cars.
The color space of YCrCb turned out to fairly separate pockets of space for the car colors. Also for non-car images the pixels are more scattered.
The following images show the color space graph:

![alt text][image1] ![alt text][image2]

![alt text][image3] ![alt text][image4]

![alt text][image5] ![alt text][image6]

#Color Histogram

I checked the color histogram of car and non-car images with YCrCb color space.
The histogram plots augment the feature set to classify the car and non-car images
I am using np.histogram() function to store the color histogram features in color_hist() function in cell no. 2.
I am using 16 bins for histogram of each channel of YCrCb converted images.
The following are the examples histogram plots:

![alt text][image10]

#Spatial Binning

Also checked with spatial binning feature.
As explained in the lesson, resized images to (16x16) is still recognizable for the human eye.
Using cv2.resize().ravel(), I am extracting the spatial binning feature in bin_spatial() in cell no. 2.
The relavent features are still retained in this feature which could be used for classification.

![alt text][image7] ![alt text][image8]

The following is the spatial bin feature vector:

![alt text][image9]

#HOG Features

HOG feature helps in giving a signature for the shape of objects which are color and scale invariant.
As explained in the lessons, it definitely helps in classfying the shapes of car.
I used the hog() function from skimage.feature to extract hog features.
The hog features are getting extracted in the function get_hog_features() in cell no. 2.
This is is used separately for each channel if YCbCr converted image.
The following images show the HOG features of a few car images.

![alt text][image11]
![alt text][image12]
![alt text][image13]

I am using the following parameters for extracting hog features:

* orient = 9
* pix_per_cell = 8 
* cell_per_block = 2 

###Classification method

For classification the following steps are executed:

* Extract features from dataset
* Normalize the feature set
* Split the dataset for training and testing and randomly shuffling
* Training the classifier and find the test accuracy with test set

The following features are extracted using a single function extract_features() in cell no. 3. I concatenate them into a single array using np.hstack():

* Spatial binning
* Color Histogram
* HOG feature

Following are the parameters:

* color_conv = 'RGB2YCrCb'
* orient = 9  # HOG orientations
* pix_per_cell = 8 # HOG pixels per cell
* cell_per_block = 2 # HOG cells per block
* spatial_size = (16, 16) # Spatial binning dimensions
* hist_bins = 16

It is observed that the YCrCb is a good fit for detecting cars. And Y channel which is the greyscale channel, would be able to capture the gradients/edges decently for cars. This should be giving good enough HOG features.
Also for 64x64 image, cells of 8x8 pixels are small enough to capture the change in gradients of the car edges.
No specific reason to choose 9 bins. Only thing kept in mind is that this would give equally spaced bins of 40 degrees between -180 to 180. Car shapes would exhibit specific sharp gradient edges and the 9 bin histogram should be able to give a signature for the car shape. With above set of features the car detection is working pretty decently.

Since each feature would be having different scales or range (max,min), I normalized them using StandardScaler library of sklearn.preprocessing.
The normalization is covered in find_cars() for inference and extract_features() for training.
This feature normalization is carried to avoid the biasing of any particular feature.

I am using the train_test_split() from sklearn.cross_validation library to split the data into training and testing dataset
And also shuffling the dataset before training. This is done in cell no. 5.
I am using 80:20 ratio train:test split-up.

For classifier I am using Linear SVM from sklearn.svm library.
The training is carried in cell no. 5.
The testing accuracy with the linear svm, I was able to achieve 99.2 % accuracy.

The trained model & parameters for feature extraction are stored in pickle file.
Same pickle is used at the time of car detection in the project video.

###Sliding Window Search

The sliding window approach is used in cell no. 9 in function find_cars().
Following are the specifications for the sliding window:

* Image scale factor = Downsize by 1.5
* Window size before scaling: 96x96
* Window size after scaling: 64x64
* overlap: 1 cell = 8 pixels in both x and y direction (After downscaling)
* region of interest: y_start = 385; y_end = 656

The original image is downscaled by 1.5 times.
A window size of 96x96 in the original image will be equivalent to 64x64 in the downscaled image.
As the training images are of 64x64 resolution, I am using a sliding window of 64x64 resolution.
By using small overlap, there is more chance of classifying of cars in the image.
It also helps in detecting multiple windows for the same car which boosts the confidence of detected cars
and helps in rejecting false positives.

Following algorithm is executed in find_cars() in cell no. 9 :

* Set window at the starting position in the region of interest of the scaled image
* Extract the features (HOG, color histogram & spatial bin) for this window. Normalize the features and concatenate them.
* Use the trained classifier to predict whether the window contains a car or not.
* If the window contains a car, then store the window bounding box
* Slide the window to the next position as per the step size.
* Follow this till the window reaches the end corner of region of interest.

Extracting HOG features could be an expensive operation if used separately for each window.
To accelerate this feature extraction step, I am employing the HOG Sub-Sampling Window Search.
In this method, the HOG features are calculated for the entire image just once for each channel.
hog() from skimage.feature is used with feature_vec=False. 
This will return the hog feature image with the same resolution as that of the input image instead of returning a flattened array.
From this hog feature image, the hog features are extracted for all the cells present in a particular window.
This will give the hog features for that particular window.
Similarly hog features can be extracted for each window by sliding it in the region of intereset of the hog feature image.
This process is executed in find_cars() in cell no.9.

Training with more dataset, improved the accuracy of the classifier. This helped in detecting more windows for the same car
and reduced the false positive detection.

Following is an example image of the detected windows in the test image.

![alt text][image26] ![alt text][image14]
![alt text][image27] ![alt text][image15]
![alt text][image28] ![alt text][image16]
![alt text][image29] ![alt text][image17]

###Heat Map
As described in the lesson, heat map is generated for overlapping windows
The heat map is calculated in the following manner:

* Initialized a heat map image which is of the same resolution as the project image
* Traversed through the bounding box. With each pixel inside a bounding box, increment the pixel value by 1 in add_head().
* The region where many windows are overlapping would end up having high pixel values.
* Applied a lower threshold the resultant heat map image, and allow only pixels values only greated than the threshold in apply_threshold()
* Used label function from scipy.ndimage.measurements library to label unique hot regions in the heat map.
* Based on labeled hot regions a tight bounding box is computed in draw_labeled_box()
* Rendered the bounding box using the cv2.rectangle()

add_head(), apply_threshold & draw_labeled_box() functions are defined in cell no.8.
The following show the heat map:

![alt text][image14] ![alt text][image18]
![alt text][image15] ![alt text][image19]
![alt text][image16] ![alt text][image20]
![alt text][image17] ![alt text][image21]

The following images show the final bounding boxes detected for the test images

![alt text][image18] ![alt text][image22]
![alt text][image19] ![alt text][image23]
![alt text][image20] ![alt text][image24]
![alt text][image21] ![alt text][image25]

### Video Implementation
Please find the uploaded output_video.mp4 which is detecting the cars in the video.
The pipelines in executed in process_image() in cell no.12.

###Improvement Methods used
With the above steps I was facing the following problems:

* The bounding boxes in the output video where very jettery and wobbling accross frames.
* There were a few false positives.

To smooth out the bounding boxes, I used bounding boxes computed from the sliding window approach for the past 15 images, applied heat map on them to get the smooth bounding box for the current frame. This logic is carried out in cell no. 12 as part of process_image().
I am using collection.deque to store the bounding boxes for the past 15 frames.
I got this suggestion from this [blog post](http://jeremyshannon.com/2017/03/17/udacity-sdcnd-vehicle-detection.html)

* No. of past frames = 15
* Heat Map Threshold = 40

This smoothened the bounding boxes accross frames.
This also eliminated the false positives. This also removed the detection of cars coming on the opposite lane

Here's a [link to my video result](./out_project.mp4)

### Discussion
There is still scope for improvement. 
For the initial few frames of the car coming into the frame, the bounding box is not shown. This is because the heat map threshold is kept high. Sliding windows of different sizes would detect more car or portions of car. This could help to get bounding boxes for the initial few frames into the frame.
The classifier detects few false positives in the shadow region. Tweaking the parameters of the classifier might help to make it more ribust.
The final bounding boxes could be a more tight.

There could be issues with the pipeline where the another car is travelling at a very high speed as compared the our car.
In that case the windows detected might be less and temporal heat map might result in not detecting the car properly.

I have used Linear SVM. Perhaps deep learning approach might be more robust in car classification with minimum false positives.
