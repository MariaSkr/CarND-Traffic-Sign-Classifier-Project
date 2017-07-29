
#**Traffic Sign Recognition** 

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"



1. Writeup includes all the rubric points. Here is a link to my [project code](https://github.com/MariaSkr/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration
In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 

![alt text](https://github.com/MariaSkr/CarND-LaneLines-P1/blob/master/01-dataset.png)

It is a table structured images with a 4 parameters:
Sign's class name
Number of occurrences in Training set
Number of occurrences in Validation set
Number of occurrences in Test set
###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because in case with traffic signs, the color does not improve the actual performane.

As a second step, I did normalization of the image data.

Here is an example of a traffic sign image before and after each step of preprocessing.

![alt text](https://github.com/MariaSkr/CarND-LaneLines-P1/blob/master/02-preprocessing.png)


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, 'VALID' padding, outputs 28x28x6	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 14x14x6 				|
| Flatten    | outputs 400    									|
|Fully connected 		| outputs 120      									|
| Dropout				| keep probability = 0.75      									|
|				Fully connected		|				outputs 84								|
|		Dropout			|	keep probability = 0.75											|
|		Fully connected		|				outputs 43 logits								|
 



####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the follow global parameters:
Number of epochs = 10. Experimental way: increasing of this parameter doesn't give significant improvements.
Batch size = 128
Learning rate = 0.001
Optimizer - Adam algorithm (alternative of stochastic gradient descent). Optimizer uses backpropagation to update the network and minimize training loss.
Dropout = 0.75 (for training set only)

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were: 

training set accuracy of 0.99303

validation set accuracy of 0.94580

test set accuracy of 0.929

This solution based on modified LeNet-5 architecture. With the original LeNet-5 architecture, I've got a validation set accuracy of about 0.88.

Architecture adjustments:

Step 1: perform preprocessing (grayscale and normalization). Results for training and validation sets on epoch #10 were 0.99253 and 0.89435 that's mean overfitting.

Step 2: generate augmented training data. Results for training and validation sets on epoch #10 were 0.98437 and 0.91633 that's mean overfitting again.

Step 3: Important design choice - apply Dropout - a simple way to prevent neural networks from overfitting). Results for training and validation sets on epoch #10 were around 0.99237 and 0.93673 (keep_prob values were in range 0.5-0.8).

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 8 German traffic signs that I found on the web:

![alt text](https://github.com/MariaSkr/CarND-Traffic-Sign-Classifier-Project/blob/master/1.jpg)
![alt text](https://github.com/MariaSkr/CarND-Traffic-Sign-Classifier-Project/blob/master/2.jpg)
![alt text](https://github.com/MariaSkr/CarND-Traffic-Sign-Classifier-Project/blob/master/3.jpg)
![alt text](https://github.com/MariaSkr/CarND-Traffic-Sign-Classifier-Project/blob/master/4.jpg)
![alt text](https://github.com/MariaSkr/CarND-Traffic-Sign-Classifier-Project/blob/master/5.jpg)
![alt text](https://github.com/MariaSkr/CarND-Traffic-Sign-Classifier-Project/blob/master/6.jpg)
![alt text](https://github.com/MariaSkr/CarND-Traffic-Sign-Classifier-Project/blob/master/7.jpg)
![alt text](https://github.com/MariaSkr/CarND-Traffic-Sign-Classifier-Project/blob/master/8.jpg)

The images might be difficult to classify because of noizy background.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here is example of a traffic sign after preprocessing:


![alt text](https://github.com/MariaSkr/CarND-Traffic-Sign-Classifier-Project/blob/master/04-web-dataset-preprocessing.png)


| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| General caution     		| General caution  									| 
| No passing   			| No passing										|
|Speed limit (20km/h)					| Speed limit (20km/h)										|
| Stop     		|Stop				 				|
| Priority road			| Priority road  							|
| Yield		| Yield							|
| Turn right ahead	|Turn right ahead				|

The model was able to correctly guess 7 of the 8 traffic signs, which gives an accuracy of 87.5%. This compares favorably to the accuracy on the test set of 92.6%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a road work sign (probability of 0.72998), but the image does contain a General caution sign. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .72998         			| Road work   									| 
| .14959     				| Road narrows on the right										|
| .06162 					| General caution sign											|
| .04716      			| Pedestrians					 				|
| .00731			    | Dangerous curve to the right     							|


For the second image,the model is relatively sure that this is a no passing sign (probability of 0.97063 ), and the image does contain a No passing sign. The top five soft max probabilities were:


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .97063        			| No passing   									| 
| .02936     				| No passing for vehicles over 3.5 metric tone										|
| 0					| Vehicles over 3.5 metric tons prohibited										|
| 0     			| No vehicles			 				|
| 0		    | Priority road    							|

For the third image,the model is relatively sure that this is a no passing sign (probability of 0.97063 ), and the image does contain a No passing sign. The top five soft max probabilities were:


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .97063        			| No passing   									| 
| .02936     				| No passing for vehicles over 3.5 metric tone										|
| 0					| Vehicles over 3.5 metric tons prohibited										|
| 0     			| No vehicles			 				|
| 0		    | Priority road    |

For the fourth image,the model is relatively sure that this is a priority road (probability of 1.0 ), and the image does contain a Priority road sign. The top five soft max probabilities were:


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0        			| Priority road   									| 
| 0     				| Road work								|
| 0					| Yield										|
| 0     			| No vehicles			 				|
| 0		    | Speed limit (30km/h)    |
### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


