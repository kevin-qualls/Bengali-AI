## Midway Blog Post: Testing Different Approaches (Mar. 09, 2020)

## Introduction
For this blog post, we have implemented some changes in preprocessing the data and tried several incremental changes to our baseline neural network architecture. Instead of jumping right into some of the high-grade CNN architectures available online, we wanted to build up a decently performing model from scratch, and then use one or two fancier tricks to improve the accuracy.

In order to preprocess the data more efficiently, we have started using the ImageDataGenerator class, which allowed us to load the dataset "in real time" while training, hence avoiding overloading the working memory. The dataset is initially available in four .parquet files, each containing around 5000 training images. We loaded the .parquet files, one piece at the time, and used them to generate and save separate image files. The ImageDataGenerator class comes with two tools for loading the data - "FlowFromDirectory" and "FlowFromDataframe" - the first one requiring images of different classes to be stored in separate folders and the second one taking in a pandas dataframe containing the file names and corresponding labels. We opted for the second one, implementing which turned out to be significantly simpler. We took advantage of the methods for rescaling the data and splitting it into the training (80%) and validation (20%).

Since finishing our last blog post, we have realized that increasing the number of filters in convolutional layers can significantly improve the performance. However, when more than 25 filters were used, this resulted in significant overfitting. This is illustrated below, we can see the validation accuracy plateauing as the training accuracy is steadily improving.

<p align="center">

<img width="468" alt="Fig 8 graph" src="https://user-images.githubusercontent.com/54907300/81539904-e657f180-933e-11ea-9b6a-72fc859c6588.png">
  
</p>
<p align="center">
  <b>Fig. 8: Comparing Validation (Blue) and Training (Orange) - Overfitting occurs at 50% mark%</b><br>
</p>


Here, the model was trained for 50 epochs. The x-axis is labeled according to epochs/50, the orange plot corresponds to the training and blue plot to the validation accuracy. One counterintuitive aspect of this plot and the following ones is that, in the initial stages of the training, the validation accuracy is actually significantly higher than the training accuracy. We attribute this to using dropouts (here, dropouts are applied only between the final two dense layers), which are applied only during the training and not validation.

## Three Approaches to Regularizing the Model

We have attempted to regularize the model by introducing dropouts after max pooling layers. After adding a dropout of 0.5 after each max pooling layer, the model performed significantly worse, with accuracy hovering around 2.5% after 5 epochs - this value was too high. We tuned both the number of filters and the dropout parameter by training the model for several epochs and choosing the best - performing model, with 20 filters, dropouts of 0.1 after max pooling layers and a dropout of 0.2 between the two dense layers at the very end of the neural network. This gave us the validation accuracy of around 47% and the training is visualized in the Tensorboard's graph below (similar to value accuracy)

<p align="center">
<img width="419" alt="Fig 9 graph" src="https://user-images.githubusercontent.com/54907300/81539906-e6f08800-933e-11ea-820f-774bdf913559.png">
</p>

<p align="center">
  <b>Fig. 9: Validation (Blue) and Training (Orange) Improve After Tuning Model</b><br>
</p>

Here, the x-axis corresponds to the number of epochs trained, the orange plot is the training and blue plot validation accuracy.

For our second approach, we tried to regularize the neural net using SpatialDropouts - a technique that drops 2D Feature maps - while simultaneously adding "l1" and "l2" regularizers to the convolution 2D layers. Between convolution layers, we added the following,

```
keras.layers.Conv2D(filters=25, kernel_size=2, activation='relu', padding="SAME",
                kernel_regularizer=regularizers.l1(0.01)),
tf.keras.layers.SpatialDropout2D(rate = 0.2, data_format=None),
```
<p align="center">
  <b>Fig. 10: Introducing Spatial Dropouts to the Model</b><br>
</p>

For our third approach, we removed SpatialDropouts and kept the regularizers consistent ("l1"). 

After performing the tuning, both the second and the third approach resulted in slightly lower omptimal accuracy (between 2.5% and 3%).

## Making the Neural Network Deeper

After optimizing the number of layers, the next logical step was to try to make the neural network deeper. As we will discuss in this section, this resulted in a large drop in accuracy even after regularizing the layers.

<!-- Our final code had 23 layers (5 Convolution 2D, 3 Dense, 4 Dropout, 1 Flatten, 4 MaxPooling, 6 SpatialDropout2D). The Convultion  Layers had filters set to 25, kernel sizes of 3, "relu" activation functions, "SAME" paddings, and regularizers set to 0.01 (3 L1 reglarizers and 2 L2 regularizers). The SpatialDropout2D layers had rates of 0.2, the MaxPooling2D layers had pool_sizes set to 2, and the Desnse layers had units of 168, kernel_initializers set to "glorot_normal", 2 activations set to "relu" and 1 set to "softmax" and regularizes set to 0.01 (1 L1 regularizer and 2 L2 regularizers). -->

After running our model for 30 epochs, we got small values for training accuracy and validation accuracy, fluctating between 2.5% and 3%:

<p align="center">
<img width="539" alt="Fig 11 graph" src="https://user-images.githubusercontent.com/54907300/81545624-26bb6d80-9347-11ea-914e-e9feb7590770.png">
</p>

<p align="center">
  <b>Fig. 11: Accuracies Less than 3% after Making Neural Network Model More Dense</b><br>
</p>

<!-- We used 17 layers - 5 Convolution 2D, 3 Dense, 4 Dropouts, 1 Flatten, 4 MaxPooling (we also added 2 dropout layers), keeping all other parameters the same (i.e. filters set to 25). After running the model for 30 epochs, we surprisingly get the same accuracy and validation accuracy, less than 3%. 

<!-- <img width="499" alt="spatial dropout approach" src="https://user-images.githubusercontent.com/54907300/76159806-69b85500-60fa-11ea-9386-8836fac8e34e.png"> -->

Since our training and validation accuracy are about the same, the problem is not due to overfitting. It could possibly be due to vanishing gradients.


## Next Steps
 
While we are somewhat satisfied with getting close to 50% accuracy while distinguishing between 168 classes of grapheme roots, it is clear that there is further progress to be made. Our simple neural network architecture seems to have reached its limit, when both increasing the number of convolutional filters and increasing the number of layers lead to a decrease in performance. 

For the remainder of this project, we would like to play with adding more layers with "skip connections", as used in the ResNet architecture and described in the previous blog post. A neural network with an added block of layers and a "skip connection" should be at least as good and hopefully better, and we should see at least a small increase in accuracy.

## References
1. Keras references were consulted extensively

...
 


