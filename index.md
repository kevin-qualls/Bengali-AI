#### Stefan Stanojevic, Kevin Qualls
#### DATA 2040: Deep Learning and Advanced Topics in Data Science

Welcome to our website for our DATA 2040 Midterm Project! 

We are graduate students at Brown University performing machine learning techniques to classify the Bengali language, as part of the following [Kaggle Competition](https://www.kaggle.com/c/bengaliai-cv19). 

For a presentation of our process and results, check out the following screencast (link), [Github Repository](https://github.com/stefs92/Bengali-AI.git) or blogs below!


## Initial Blog Post: Assessing the Challenge (Feb. 18, 2020)

## Introduction
While being spoken by more than 200 milion people, Bengali language is particulary interesting from the point of view of AI handwritten recognition. Each bengali letter consists of 3 parts -one of 168 possible grapheme roots, one of 11 possible vowel diacritics and one of 7 possible consonant diacritics. The sheer number of combinations makes handwritten symbol recognition a challenging machine learning problem.


At a high level, we wish to break down an image of a Bengali word and assign the pieces to three bins:
<p align="center">
<img width="647" alt="high_level_picture" src="https://user-images.githubusercontent.com/54907300/74720359-abdd2d80-5203-11ea-90a5-734785bae48b.png">
  
</p>
<p align="center">
  <b>Fig. 1: High-level Description of Project Task</b><br>
</p>

Although it's a steep task, our team is prepared and has prior experience with image classification that could be helpful, such as working with the renowned MNIST Dataset (shown below) to organize numbers by different fonts:

<p align="center">
<img width="575" alt="Screen Shot 2020-02-18 at 1 47 18 AM" src="https://user-images.githubusercontent.com/54907300/74720496-e941bb00-5203-11ea-9626-bfdd9d10ecb4.png">
 </p>

## Examining the Data

When loading the data, we see there are approximately 10,000 grapheme images to work with. 

We will mostly be using the .parquet train and test files, each of which contains tens of thousands of images (each size 137 x 236). They are easily loaded with the help from pandas package. Each row represents an image, and we plotted one row as a trial run:

<p align="center">
<img width="436" alt="Screen Shot 2020-02-18 at 5 26 45 AM" src="https://user-images.githubusercontent.com/54907300/74727573-6292db00-520f-11ea-8242-8b36604e1408.png">
</p>

We noticed the image has some similarities to the 94th grapheme root from the glossary:

<p align="center">
<img width="329" alt="map_94" src="https://user-images.githubusercontent.com/54907300/74727346-04fe8e80-520f-11ea-9693-86e82d1ed432.png">
 </p>

We believe the more we manually look at the images, the more we can improve our understanding of the Bengali language, which can ultimately help us form our model. 

## Neural Network Model: the Grapheme Root

As an initial step, we decided to focus on a simpler problem: to design a Neural Network capable of recognizing the grapheme root. 
We choose to do so in order to quickly have a working model and begin to assess the difficulties of the task. Recognizing the grapheme root provides the most difficult step since it involves 168 different classes compared to the 7 and 11 of the diacritics components. 

In addition, the model trained in recognizing the grapheme root can then be used to tackle the entire classification problem, for example by adding layers to the network which will be trained to recognize the diacritics.
Having in mind that the diacritics are essentially decorations of the grapheme root, it seems reasonable that an effective neural network should work by first recognizing the root and consequently any extra addition to it. 


We took as a starting point a simple convolutional neural network taken from from Geron's companion notebook to Chapter 14 of Hands-On Machine Learning, which we suitably tuned to our problem.


```markdown
heigth = 137;
width = 236;

model = keras.models.Sequential([
    keras.layers.MaxPooling2D(pool_size=2,input_shape=[heigth, width, 1]),
    keras.layers.Conv2D(filters=8, kernel_size=7, activation='relu', padding="SAME"),
    keras.layers.MaxPooling2D(pool_size=3),
    keras.layers.Conv2D(filters=8, kernel_size=3, activation='relu', padding="SAME"),
    keras.layers.Conv2D(filters=8, kernel_size=3, activation='relu', padding="SAME"),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Conv2D(filters=8, kernel_size=3, activation='relu', padding="SAME"),
    keras.layers.Conv2D(filters=8, kernel_size=3, activation='relu', padding="SAME"),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dense(units=168, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=168, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=168, activation='softmax'),
   
```



We then started training the network on a portion of the available training data: the 50.000 images contained in the file train_image_data_0.parquet available at https://www.kaggle.com/c/bengaliai-cv19/data.

By a few trial and errors we have figured out a good initial set of hyperparameters (pooling sized and number of filters) for our neural network, obtaining a validation accuracy of 41% after 50 epochs of training. Considering that we have 168 classes, we can see that a random guessing would give an accuracy of approximately 0.5% instead. We used TensorBoard to visualize the training process. Here's a snapshot of the validation accuracy as a function of the number of epochs.


<p align="center">
<img width="375" alt="accuracy" src="https://user-images.githubusercontent.com/54907300/74803236-26f91f00-52aa-11ea-85c4-f46c7275a226.png">
</p>

## Next Steps

We noticed the images we loaded have a large yellow cloud around the graphemes. To prevent the model from unnecessarily traning this yellow space, we hope to focus the model on just the blue-lined grapheme. This would involve looking at bounding boxes for our images. Cropping to the union of bounding boxes for all images would be a safe bet but might still result in unnecessarily large image size - therefore, we might want to restrict to a box size large enough to cover (100 - p)% of the images where p is a small parameter that can be tuned to increase accuracy.

We can also experiment with different possible ways of training the network. The full dataset seems too large for Keras to handle simultaneously, so one way to train would to be to split it into 4 pieces and train on each one separately for some number_of_epochs. It is possible that accuracy would increase if we reduce number_of_epochs and then repat the process many times - this way, the neural network would have a chance to look at the entire dataset before getting really good at predicting its subsets. We can also try to use an instance of the ImageDataGenerator class in order to load objects in real time while training.

We also would like to find an effective way to normalize the data. We initially tried dividing the data by its max image size (255), however the program crashes when doing so, perhaps since floats take up more memory than integers and we are already pushing RAM to its limits due to sheer amount of data. Using the ImageDataGenerator class could fix this issue as well.

Most importantly, we will experiment more with different neural network architectures and look for inspiration within the publicly available high-grade convolutional neural networks, and from the rich body of literature available on this topic. When faced with the problem of designing an efficient neural network architecture, one's first instinct is to add more layers. However, this leads to two issues that are really two sides of the same coin - increased computational complexity of training and overfitting. As noted in the famous ResNet paper, it is even common for training accuracy of overly deep models to decrease, a problem beyond overfitting. Their proposed solution is to add an identity function to the output of blocks of layers in their neural network, like in the below figure taken from the paper.

<p align="center">
<img width="280" alt="relu" src="https://user-images.githubusercontent.com/54907300/74802429-d84a8580-52a7-11ea-8cdc-dd00f6a806af.png">
</p>

The idea is to allow deeper neural networks to more easily approximate shallower ones - in order for above block to "disappear", it is enough to set all weights and biases to zero in the above block. If we choose to make our network particularly deep, we would like to incorporate this kind of structures to help with training.

Finally, while ideally we would like to take a shot at designing our own neural network from scratch, we can also try to apply transfer learning - take examples of public source high - performing convolutional neural networks from the internet, and retrain the last couple of layers to adapt them to our task.

## References

[1] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). doi: 10.1109/cvpr.2016.90


...


## Midway Blog Post: Testing Different Approaches (Mar. 09, 2020)

## Introduction
For this blog post, we have implemented some changes in preprocessing the data and tried several incremental changes to our baseline neural network architecture. Instead of jumping right into some of the high-grade CNN architectures available online, we wanted to build up a decently performing model from scratch, and then use one or two fancier tricks to improve the accuracy.

In order to preprocess the data more efficiently, we have started using the ImageDataGenerator class, which allowed us to load the dataset "in real time" while training, hence avoiding overloading the working memory. The dataset is initially available in four .parquet files, each containing around 5000 training images. We loaded the .parquet files, one piece at the time, and used them to generate and save separate image files. The ImageDataGenerator class comes with two tools for loading the data - "FlowFromDirectory" and "FlowFromDataframe" - the first one requiring images of different classes to be stored in separate folders and the second one taking in a pandas dataframe containing the file names and corresponding labels. We opted for the second one, implementing which turned out to be significantly simpler. We took advantage of the methods for rescaling the data and splitting it into the training (80%) and validation (20%).

Since finishing our last blog post, we have realized that increasing the number of filters in convolutional layers can significantly improve the performance. However, when more than 25 filters were used, this resulted in significant overfitting. This is illustrated below, we can see the validation accuracy plateauing as the training accuracy is steadily improving.

<p align="center">
<img width="400" alt="high_level_picture" src="https://user-images.githubusercontent.com/31740043/76132697-35ea0c00-5fe2-11ea-881e-02bda7e403ba.PNG">
</p>

Here, the model was trained for 50 epochs. The x-axis is labeled according to epochs/50, the orange plot corresponds to the training and blue plot to the validation accuracy. One counterintuitive aspect of this plot and the following ones is that, in the initial stages of the training, the validation accuracy is actually significantly higher than the training accuracy. We attribute this to using dropouts (here, dropouts are applied only between the final two dense layers), which are applied only during the training and not validation.

# Three Approaches to Regularizing the Model

We have attempted to regularize the model by introducing dropouts after max pooling layers. After adding a dropout of 0.5 after each max pooling layer, the model performed significantly worse, with accuracy hovering around 2.5% after 5 epochs - this value was too high. We tuned both the number of filters and the dropout parameter by training the model for several epochs and choosing the best - performing model, with 20 filters, dropouts of 0.1 after max pooling layers and a dropout of 0.2 between the two dense layers at the very end of the neural network. This gave us the validation accuracy of around 47% and the training is visualized in the Tensorboard's graph below (similar to value accuracy)

<p align="center">
<img width="400" alt="high_level_picture" src="https://user-images.githubusercontent.com/31740043/76132939-82821700-5fe3-11ea-90cb-9e39500aff20.PNG">
</p>

Here, the x-axis corresponds to the number of epochs trained, the orange plot is the training and blue plot validation accuracy.

For our second approach, we tried to regularize the neural net using SpatialDropouts - a technique that drops 2D Feature maps - while simultaneously adding "l1" and "l2" regularizers to the convolution 2D layers. Between convolution layers, we added the following,

```
keras.layers.Conv2D(filters=25, kernel_size=2, activation='relu', padding="SAME",
                kernel_regularizer=regularizers.l1(0.01)),
tf.keras.layers.SpatialDropout2D(rate = 0.2, data_format=None),
```

For our third approach, we removed SpatialDropouts and kept the regularizers consistent ("l1"). 

After performing the tuning, both the second and the third approach resulted in slightly lower omptimal accuracy (between 2.5% and 3%).

## Making the Neural Network Deeper

After optimizing the number of layers, the next logical step was to try to make the neural network deeper. As we will discuss in this section, this resulted in a large drop in accuracy even after regularizing the layers.

<!-- Our final code had 23 layers (5 Convolution 2D, 3 Dense, 4 Dropout, 1 Flatten, 4 MaxPooling, 6 SpatialDropout2D). The Convultion  Layers had filters set to 25, kernel sizes of 3, "relu" activation functions, "SAME" paddings, and regularizers set to 0.01 (3 L1 reglarizers and 2 L2 regularizers). The SpatialDropout2D layers had rates of 0.2, the MaxPooling2D layers had pool_sizes set to 2, and the Desnse layers had units of 168, kernel_initializers set to "glorot_normal", 2 activations set to "relu" and 1 set to "softmax" and regularizes set to 0.01 (1 L1 regularizer and 2 L2 regularizers). -->

After running our model for 30 epochs, we got small values for accuracy and validation accuracy, fluctating between 2.5% and 3%:

<p align="center">
<img width="499" alt="spatial dropout approach" src="https://user-images.githubusercontent.com/54907300/76158467-72a22a00-60ec-11ea-9028-eaf247832c72.png">
</p>

<!-- We used 17 layers - 5 Convolution 2D, 3 Dense, 4 Dropouts, 1 Flatten, 4 MaxPooling (we also added 2 dropout layers), keeping all other parameters the same (i.e. filters set to 25). After running the model for 30 epochs, we surprisingly get the same accuracy and validation accuracy, less than 3%. 

<!-- <img width="499" alt="spatial dropout approach" src="https://user-images.githubusercontent.com/54907300/76159806-69b85500-60fa-11ea-9386-8836fac8e34e.png"> -->

Since our training and validation accuracy are about the same, the problem is not due to overfitting. It could possibly be due to vanishing gradients.


## Next Steps
 
While we are somewhat satisfied with getting close to 50% accuracy while distinguishing between 168 classes of grapheme roots, it is clear that there is further progress to be made. Our simple neural network architecture seems to have reached its limit, when both increasing the number of convolutional filters and increasing the number of layers lead to a decrease in performance. 

For the remainder of this project, we would like to play with adding more layers with "skip connections", as used in the ResNet architecture and described in the previous blog post. A neural network with an added block of layers and a "skip connection" should be at least as good and hopefully better, and we should see at least a small increase in accuracy.

## References
[1] tf.keras.layers.SpatialDropout2D | TensorFlow Core v.2

...
 





## Final Blog Post: Final Model (Mar. 08, 2020)

## Introduction

Prior to training the final model, we found a bounding box for each image, and then cropped and resized all images to the same size of (50,100) pixels using skimage.transform library. Initially, all pixels have integer values ranging from 0 to 255, with the "empty" pixels taking values close to 255. For the purposes of finding the bounding box, we experimented with different thresholds, and found that removing rows and columns containing only values greater then 200 works pretty well. The cropped images were saved to new .parquet files using pandas.

Since using an instance of ImageDataGenerator class was sometimes causing the session to crash, in our final work we decided to revert back to loading the four cropped image datasets piecewise. We loaded and trained on datasets separately for a number of iterations, with an EarlyStopping callback function. When validation loss on a given part of the dataset would start decreasing, we would switch to another part of the dataset and repeat the process a number of times. We used 5000 images from each of the four sets for validation and the remaining roughly 45000 for training.

## Model

We have previously noticed that our model started to perform significantly worse when additional convolutional layers were added. For our final model, we have attempted to get some additional performance by making our neural network 3 layers deeper while introducing a ResNet - style connection short-circuiting the additional layers. Our implementation of the residual block was inspired by [1] and the example from our last homework, in which we learned how to customize Keras models using a bit of Tensorflow backend. Since our convolutions had stride = 1 and 'same' padding, the dimension of their output was the same as the input dimension (except the first convolution, which introduced a number of filters). This fact made combining the two tensors at the end of the residual block particularly simple.

The summary of our final model is

<p align="center">
<img width="400" alt="high_level_picture" src="https://user-images.githubusercontent.com/31740043/76186045-81014c00-61a7-11ea-8a84-ec434d9ccf7a.png">
</p>

Where all convolutional layers had 20 filters of dimensions 3x3 and relu activations. Prior to writing the midway blog post, we performed extensive tuning of the number of filters; after exploring several different values with our new model, we found that the same values produced the best performance this time as well.

Since we were loading the data piecewise while training so as not to overload the working memory, the performance graphs are pieced together from different training sessions. In order to do so, we used Tensorboard's "Wall" option. The performance of our final model on predicting grapheme roots is shown below,

<p align="center">
<img width="400" alt="high_level_picture" src="https://user-images.githubusercontent.com/31740043/76186411-9cb92200-61a8-11ea-9681-ce45de4d7569.PNG">
</p>

<p align="center">
<img width="400" alt="high_level_picture" src="https://user-images.githubusercontent.com/31740043/76189478-3258af80-61b1-11ea-9017-3aab3f3d04d3.PNG">
</p>

Where the x-axis corresponds to training epochs and two different plots correspond to training and validation accuracies. The whole process took around 40 epochs. During the majority of the training process, our training accuracy was actually lagging behind the validation accuracy. We believe that this is due to the fact that our model contains dropouts, which are only used for training and not for testing and validation. Our validation accuracy for grapheme roots ended up hovering around 70%. We were hoping to do better, but this still seems decent for a problem with 168 classes.

After training on the grapheme roots, we replaced the last layer of the model with the one appropriate for predicting vowel and consonant diacritics (with 11 and 7 outputs, respectively). Training the neural network on vowel diacritics resulted in the following performance graph, 

<p align="center">
<img width="400" alt="high_level_picture" src="https://user-images.githubusercontent.com/31740043/76188824-806cb380-61af-11ea-9774-82076c202865.PNG">
</p>

where our final validation accuracy ended up being around 90%. In this plot, validation accuracy plots are again higher than the ones for training accuracy.

Training our model on consonant diacritics instead gave us the accuracy of around 93% and resulted in the following piecewise performance graph,

<p align="center">
<img width="400" alt="high_level_picture" src="https://user-images.githubusercontent.com/31740043/76189139-51a30d00-61b0-11ea-9fa7-ab741628c8dd.PNG">
</p>

In the first two plots on the left, validation accuracy is higher than the training accuracy. Then, in the third plot, the training accuracy starts smaller but overtakes the validation accuracy, signaling that some overfitting is starting to take place.

Our Kaggle submission is available [here](https://www.kaggle.com/stefanstanojevic/kernel2b55603361?scriptVersionId=30126084), and resulted in a weighted test accuracy of 75.38%.

## Future Work

A simple possible imporovement to explore would be preprocessing the images more efficiently. After cropping the images, we have resized all of them to be of the same shape. One thing we noticed while looking at some of the images was that the aspect ratios of cropped graphemes vary widely, as images range from horizontal to vertical. It would be interesting to explore whether different kinds of cropping/resizing could incresase the accuracy a bit.

It is also possible that our way of training the model is slightly suboptimal. We would load and train on four parts of the dataset separately until the validation loss would start to increase. Training on each part of the dataset for a single epoch instead could concievably lead to better performance; however, this would take more time since then we would have to load a large file each epoch.

Finally, the model itself is probably where the largest imporvements can be made. Our goal for this project was to try to build a decently performing CNN architecture starting from scratch. While certainly there are far more successful architectures available online, we consider this goal to be fulfilled considering our basic starting point.




## References

[1] Implementation of the Residual Block at https://github.com/relh/keras-residual-unit/blob/master/residual.py

[2] Keras references were consulted extensively and for help locating simple python commands, StackExchange

[3] The inspiration for starting model was taken from Geron's companion notebook

