<!-- ## Welcome to GitHub Pages -->

<!-- You can use the [editor on GitHub](https://github.com/stefs92/Bengali-AI/edit/master/index.md) to maintain and preview the content for your website in Markdown files. -->

<!-- Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files. -->

<!-- Hey guys, I commented out some lines with this command, I guess this is GitHub's way of commenting -->


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

Although it's a steep task, our team is prepared and has prior experience with image classification that could be helpful, such as working with the renowned MNIST Dataset to organize numbers by different fonts:

<p align="center">
<img width="575" alt="Screen Shot 2020-02-18 at 1 47 18 AM" src="https://user-images.githubusercontent.com/54907300/74720496-e941bb00-5203-11ea-9626-bfdd9d10ecb4.png">
 
</p>
<p align="center">
  <b>Fig. 2: MNIST Dataset - Classifying Numeric Images. Image adapted from Wikipedia [1].</b><br>
</p>
 
## Examining the Data

When loading the data, we see there are approximately 10,000 grapheme images to work with. 

We will mostly be using the .parquet train and test files, each of which contains tens of thousands of images (each size 137 x 236). They are easily loaded with the help from pandas package. Each row represents an image, and we plotted one row as a trial run:

<p align="center">
<img width="436" alt="Screen Shot 2020-02-18 at 5 26 45 AM" src="https://user-images.githubusercontent.com/54907300/74727573-6292db00-520f-11ea-8242-8b36604e1408.png">

</p>
<p align="center">
  <b>Fig. 3: Plotting a Graphemic Root as a Trial Run</b><br>
</p>

We noticed the image has some similarities to the 94th grapheme root from the glossary:

<p align="center">
<img width="329" alt="map_94" src="https://user-images.githubusercontent.com/54907300/74727346-04fe8e80-520f-11ea-9693-86e82d1ed432.png">

</p>
<p align="center">
  <b>Fig. 4: 94th Grapheme Root</b><br>
</p>
 
 

We believe the more we manually look at the images, the more we can improve our understanding of the Bengali language, which can ultimately help us form our model. 

## Neural Network Model: the Grapheme Root

As an initial step, we decided to focus on a simpler problem: to design a Neural Network capable of recognizing the grapheme root. 
We choose to do so in order to quickly have a working model and begin to assess the difficulties of the task. Recognizing the grapheme root provides the most difficult step since it involves 168 different classes compared to the 7 and 11 of the diacritics components. 

In addition, the model trained in recognizing the grapheme root can then be used to tackle the entire classification problem, for example by adding layers to the network which will be trained to recognize the diacritics.
Having in mind that the diacritics are essentially decorations of the grapheme root, it seems reasonable that an effective neural network should work by first recognizing the root and consequently any extra addition to it. 


We took as a starting point a simple convolutional neural network taken from Geron's companion notebook to Chapter 14 of Hands-On Machine Learning, which we suitably tuned to our problem.


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

<p align="center">
  <b>Fig. 5: Simple Convolutional Neural Network</b><br>
</p>


We then started training the network on a portion of the available training data: the 50.000 images contained in the file train_image_data_0.parquet available at https://www.kaggle.com/c/bengaliai-cv19/data.

By a few trial and errors we have figured out a good initial set of hyperparameters (pooling sized and number of filters) for our neural network, obtaining a validation accuracy of 41% after 50 epochs of training. Considering that we have 168 classes, we can see that a random guessing would give an accuracy of approximately 0.5% instead. We used TensorBoard to visualize the training process. Here's a snapshot of the validation accuracy as a function of the number of epochs.


<p align="center">
<img width="433" alt="Fig 6 graph" src="https://user-images.githubusercontent.com/54907300/81533852-f1a61f80-9334-11ea-9b68-1f25cf2c5499.png">

</p>
<p align="center">
  <b>Fig. 6: Model Validation Accuracy of 41%</b><br>
</p>



## Next Steps

We noticed the images we loaded have a large yellow cloud around the graphemes. To prevent the model from unnecessarily traning this yellow space, we hope to focus the model on just the blue-lined grapheme. This would involve looking at bounding boxes for our images. Cropping to the union of bounding boxes for all images would be a safe bet but might still result in unnecessarily large image size - therefore, we might want to restrict to a box size large enough to cover (100 - p)% of the images where p is a small parameter that can be tuned to increase accuracy.

We can also experiment with different possible ways of training the network. The full dataset seems too large for Keras to handle simultaneously, so one way to train would to be to split it into 4 pieces and train on each one separately for some number_of_epochs. It is possible that accuracy would increase if we reduce number_of_epochs and then repat the process many times - this way, the neural network would have a chance to look at the entire dataset before getting really good at predicting its subsets. We can also try to use an instance of the ImageDataGenerator class in order to load objects in real time while training.

We also would like to find an effective way to normalize the data. We initially tried dividing the data by its max image size (255), however the program crashes when doing so, perhaps since floats take up more memory than integers and we are already pushing RAM to its limits due to sheer amount of data. Using the ImageDataGenerator class could fix this issue as well.

Most importantly, we will experiment more with different neural network architectures and look for inspiration within the publicly available high-grade convolutional neural networks, and from the rich body of literature available on this topic. When faced with the problem of designing an efficient neural network architecture, one's first instinct is to add more layers. However, this leads to two issues that are really two sides of the same coin - increased computational complexity of training and overfitting. As noted in the famous ResNet paper, it is even common for training accuracy of overly deep models to decrease, a problem beyond overfitting. Their proposed solution is to add an identity function to the output of blocks of layers in their neural network (1), like in the below figure taken from the paper.

<p align="center">
<img width="280" alt="relu" src="https://user-images.githubusercontent.com/54907300/74802429-d84a8580-52a7-11ea-8cdc-dd00f6a806af.png">
  
</p>
<p align="center">
  <b>Fig. 7: Identity Layers: A Solution to Increase Training Accuracy%</b><br>
</p>


The idea is to allow deeper neural networks to more easily approximate shallower ones - in order for above block to "disappear", it is enough to set all weights and biases to zero in the above block. If we choose to make our network particularly deep, we would like to incorporate this kind of structures to help with training.

Finally, while ideally we would like to take a shot at designing our own neural network from scratch, we can also try to apply transfer learning - take examples of public source high - performing convolutional neural networks from the internet, and retrain the last couple of layers to adapt them to our task.

## References

(1) He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). doi: 10.1109/cvpr.2016.90

## Image References
[1] MNIST database - Wikipedia - https://en.wikipedia.org/wiki/MNIST_database

...
