## Final Blog Post: Final Model (Mar. 08, 2020)

## Introduction

Prior to training the final model, we found a bounding box for each image, and then cropped and resized all images to the same size of (50,100) pixels using skimage.transform library. Initially, all pixels have integer values ranging from 0 to 255, with the "empty" pixels taking values close to 255. For the purposes of finding the bounding box, we experimented with different thresholds, and found that removing rows and columns containing only values greater then 200 works pretty well. The cropped images were saved to new .parquet files using pandas.

Since using an instance of ImageDataGenerator class was sometimes causing the session to crash, in our final work we decided to revert back to loading the four cropped image datasets piecewise. We loaded and trained on datasets separately for a number of iterations, with an EarlyStopping callback function. When validation loss on a given part of the dataset would start decreasing, we would switch to another part of the dataset and repeat the process a number of times. We used 5000 images from each of the four sets for validation and the remaining roughly 45000 for training.

## Model

We have previously noticed that our model started to perform significantly worse when additional convolutional layers were added. For our final model, we have attempted to get some additional performance by making our neural network 3 layers deeper while introducing a ResNet - style connection short-circuiting the additional layers. Our implementation of the residual block was inspired by [1] and the example from our last homework, in which we learned how to customize Keras models using a bit of Tensorflow backend. Since our convolutions had stride = 1 and 'same' padding, the dimension of their output was the same as the input dimension (except the first convolution, which introduced a number of filters). This fact made combining the two tensors at the end of the residual block particularly simple.

The summary of our final model is:

<p align="center">
<img width="400" alt="high_level_picture" src="https://user-images.githubusercontent.com/31740043/76186045-81014c00-61a7-11ea-8a84-ec434d9ccf7a.png">
</p>

<p align="center">
  <b>Fig. 12: Final Model Schematic</b><br>
</p>

Where all convolutional layers had 20 filters of dimensions 3x3 and relu activations. Prior to writing the midway blog post, we performed extensive tuning of the number of filters; after exploring several different values with our new model, we found that the same values produced the best performance this time as well.

Since we were loading the data piecewise while training so as not to overload the working memory, the performance graphs are pieced together from different training sessions. In order to do so, we used Tensorboard's "Wall" option. The performance of our final model on predicting grapheme roots is shown below,

<p align="center">
<img width="367" alt="Fig 13 Graph" src="https://user-images.githubusercontent.com/54907300/81545623-2622d700-9347-11ea-9fb3-bf07ae5b4dd2.png">
</p>

<p align="center">
  <b>Fig. 13: Final Model Accuracy and Loss for Grapheme Roots</b><br>
</p>


Where the x-axis corresponds to training epochs and two different plots correspond to training and validation accuracies. The whole process took around 40 epochs. During the majority of the training process, our training accuracy was actually lagging behind the validation accuracy. We believe that this is due to the fact that our model contains dropouts, which are only used for training and not for testing and validation. Our validation accuracy for grapheme roots ended up hovering around 70%. We were hoping to do better, but this still seems decent for a problem with 168 classes.

After training on the grapheme roots, we replaced the last layer of the model with the one appropriate for predicting vowel and consonant diacritics (with 11 and 7 outputs, respectively). Training the neural network on vowel diacritics resulted in the following performance graph, 

<p align="center">
<img width="445" alt="Fig 14 " src="https://user-images.githubusercontent.com/54907300/81545622-2622d700-9347-11ea-9c36-04c9c2dbbd47.png">
</p>

<p align="center">
  <b>Fig. 14: Final Model Accuracy for Vowel Diacritics</b><br>
</p>

where our final validation accuracy ended up being around 90%. In this plot, validation accuracy plots are again higher than the ones for training accuracy.

Training our model on consonant diacritics instead gave us the accuracy of around 93% and resulted in the following piecewise performance graph,

<p align="center">
<img width="384" alt="Fig 15" src="https://user-images.githubusercontent.com/54907300/81545620-258a4080-9347-11ea-843f-061d23bcafa5.png">
</p>

<p align="center">
  <b>Fig. 15: Final Model Accuracy for Consonant Diacritics</b><br>
</p>


In the first two plots on the left, validation accuracy is higher than the training accuracy. Then, in the third plot, the training accuracy starts smaller but overtakes the validation accuracy, signaling that some overfitting is starting to take place.

Our Kaggle submission is available [here](https://www.kaggle.com/stefanstanojevic/kernel2b55603361?scriptVersionId=30126084), and resulted in a weighted test accuracy of 75.38%.

## Future Work

A simple possible imporovement to explore would be preprocessing the images more efficiently. After cropping the images, we have resized all of them to be of the same shape. One thing we noticed while looking at some of the images was that the aspect ratios of cropped graphemes vary widely, as images range from horizontal to vertical. It would be interesting to explore whether different kinds of cropping/resizing could incresase the accuracy a bit.

It is also possible that our way of training the model is slightly suboptimal. We would load and train on four parts of the dataset separately until the validation loss would start to increase. Training on each part of the dataset for a single epoch instead could concievably lead to better performance; however, this would take more time since then we would have to load a large file each epoch.

Finally, the model itself is probably where the largest imporvements can be made. Our goal for this project was to try to build a decently performing CNN architecture starting from scratch. While certainly there are far more successful architectures available online, we consider this goal to be fulfilled considering our basic starting point.




## References

1. Implementation of the Residual Block at https://github.com/relh/keras-residual-unit/blob/master/residual.py

2. Keras references were consulted extensively and for help locating simple python commands, StackExchange

3. The inspiration for starting model was taken from Geron's companion notebook

