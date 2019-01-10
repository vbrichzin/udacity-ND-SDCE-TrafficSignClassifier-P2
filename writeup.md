# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./data-distributions.png        "Data Visualization"
[image2]: ./image-normalized-gray.png     "Grayscaling"

[image4]: ./examples/Vorfahrt-achten.jpg  "Traffic Sign 1"
[image5]: ./examples/Rechts-einfahren.jpg "Traffic Sign 2"
[image6]: ./examples/Kreisverkehr.jpg     "Traffic Sign 3"
[image7]: ./examples/Personen-kreuzen.jpg "Traffic Sign 4"
[image8]: ./examples/Keine-Durchfahrt.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/vbrichzin/udacity-ND-SDCE-TrafficSignClassifier-P3/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the `shape` attribute to calculate summary statistics of the traffic
signs data set:

* The size of training set is **34799**
* The size of the validation set is **4410**
* The size of test set is **12630**
* The shape of a traffic sign image is **(32,32)**
* The number of unique classes/labels in the data set is **43**

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is showing three bar charts of each the training, validation and test data.

The data shows that the distribution of the frequency in the occurrence of the signs in the datasets is similar across the classes.
It also shows that for a lot of classes above class 19 there are less pictures of it in the training set.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

I first used the tensorflow function `tf.image.convert_image_dtype` to have the data converted to an RGB float format that was needed as input to convert to grayscale with the tensorflow function `tf.image.rgb_to_grayscale`.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

I did not perform additional pre-processing on the data. Instead I tuned my model to reach the required validation threshold of 0.93.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscaled and normalized image		| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| tanh					| as suggested by Yann LeCun paper				|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| tanh                  | as suggested by Yann LeCun paper				|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				    |
| Flatten               | output 400                                    |
| Fully connected		| input 400, output 120        					|
| tanh                  | as suggested by Yann LeCun paper				|
| Fully connected		| input 120, output 84        					|
| tanh                  | as suggested by Yann LeCun paper				|
| Fully connected		| input 84, output 43        					|

I then applied the additional steps also from the LeNet example, using `tf.nn.softmax_cross_entropy_with_logits` to calculate the crossentropy and using the `tf.train.AdamOptimizer` for optimization of the loss.


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model I used a batch size of 128.
The learning rate I kept at the recommended 0.001 value.
I started with 10 epochs and increased to 20 in the end.
The stddev sigma value I started with 0.05 switched to 0.1 and came back to 0.05.

The final validation accuracy was greater than 0.93 in the end.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of **1.000**
* validation set accuracy of **0.942** 
* test set accuracy of **0.918**

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I chose the LeNet architecture. I honestly didn't really have an alternative here as it was suggested and I had understood it enough.
* What were some problems with the initial architecture?
I ran it first without grayscaling and obviously made a mistake in the normalization. It took me a while to do the normalization properly.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
Even with properly normalized and grayscaled images I only reached validation values of 0.91. In the paper by Yann LeCun the tanh was suggested as an alternative activation function. So I switched all ReLo activation functions to tanh activation functions and achieved above 0.93 for the validation set.
* Which parameters were tuned? How were they adjusted and why?
I played around with the stddev for the initial weights because my initial accuracy in the first epoch was much below the ones seen in the video which also used the LeNet architecture. I didn't find this parameter to change much so I stayed with 0.05.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
The convolution helps because it makes the model less sensitive to where exactly in the picture the features are that should be identified.
I didn't try dropout as I understood it to be helpful for computation while achieving the same results. Since I achieved 0.93 I didn't do droput even though it didn't look too complicated in the video.

If a well known architecture was chosen:
* What architecture was chosen?
LeNet from the MNIST example in the course.
* Why did you believe it would be relevant to the traffic sign application?
Well, it was suggested as a good model to use and in the end the tasks are similar to the MNIST where it worked well for the classification.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
The training accuracy is obviously very high as the model was trained with it.
The validation set wasn't as high, but since it was used in the iterative solving still higher than the test set which was above 0.90, but still the lowest of all three.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] ![alt text][image7] ![alt text][image8]

The first three images should be easy to classify as they were in a similar way in the training data set and the images are of good quality and lighting.
The fourth image (elderly pedestrians) should be difficult to identify as this wasn't in the training set. Yet children crossing and pedestrians was in the training set, so there was a chance for identifying.
The last image was unlikely identified as it is not a German traffic sign (Paddington underground station).

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Yield sign      		| Yield sign   									| 
| Keep right    		| Keep right 									|
| Roundabout mandatory	| Roundabout mandatory							|
| Elderly crossing 		| Dangerous curve to the right	 				|
| Paddington 			| 100 km/h speed limit 							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This compares roughly favorably to the accuracy on the test set of 0.91, as the fourth and fifth image were very difficult to identify as they were not in the training set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image, the model is very sure that this is a yield sign (probability of 0.999), and the image does contain a yield sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.999         		| Yield sign   									| 
| 1.11e-5  				| No passing    								|
| 1.03e-5				| Ahead only									|
| 1.18e-6      			| 60 km/h speed limit			 				|
| 9.87e-7			    | Priority road     							|

For the second image, the model is very sure that this is a keep right sign (probability of 0.989), and the image does contain a keep right sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.989         		| Keep right sign   							| 
| 6.06e-3  				| 30 km/h speed limit    						|
| 2.00e-3				| 50 km/h speed limit							|
| 9.11e-4      			| Priority road 			 					|
| 8.00e-4			    | 60 km/h speed limit     						|

For the third image, the model is very sure that this is a roundabout mandatory sign (probability of 0.999), and the image does contain a roundabout mandatory sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.999         		| Roundabout mandatory   						| 
| 6.58e-4  				| 100 km/h speed limit    						|
| 3.63e-4				| Right-of-way at the next intersection			|
| 2.74e-4      			| Priority road			 						|
| 4.98e-5			    | End of no passing by vehicles over 3.5 metric tons|

For the foruth image, the model is very sure that this is a dangerous curve to the right sign (probability of 0.995), but the image does contains a elderly pedestrian crossing. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.995         		| Dangerous curve to the right   				| 
| 4.11e-3  				| Keep right    								|
| 1.37e-4				| Slippery road									|
| 1.20e-4      			| Turn left ahead			 					|
| 1.16e-4			    | End of no passing     						|

It was very difficult for the model to correctly recognize the sign as it wasn't in the training data set, only the related and a bit similar signs of pedestrians and children crossings were in the data set. So no big surprise it wasn't correctly recognized. Probably the curved lines in the middle led to the classification, as well as for the runner up.

For the fifth image, the model is confident that this is a 100 km/h speed limit sign (probability of 0.735), but the image contains a UK underground sign for Paddington station. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.735         		| 100 km/h speed limit   						| 
| 0.111  				| 120 km/h speed limit    						|
| 0.065				    | Vehicles over 3.5 metric tons prohibited		|
| 0.040      			| Roundabout mandatory			 				|
| 0.029			        | No passing     								|

I wasn't expecting the model to classify this sign correctly. Since it does have similarities with signs where a darker circle has a lighter area in the middle and also some text, I don't find it surprising that the speed limit signs were the best candidates for the model.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


