**Microsoft Azure Scholarship Project Showcase Challenge**

## **Viral vs Bacterial Pneumonia Image Classification using Transfer Learning**

**Dataset:** Chest X-Ray Images (https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

**Team Member(s)**: Mukund 

The original problem statement is to classify normal Chest X -rays and Pneumonia Chest X-rays. The pneumonia images contain both bacterial and viral pneumonia images.

***To make the problem more challenging, the normal images were removed and the pneumonia images alone were used to build a binary classifier than can distinguish between Viral and Bacterial Pnenumonia.***

## **Business Problem**

**Existing Scenario**

Pneumonia is a life-threatening disease, which occurs in the lungs caused by either bacterial or viral infection. It can be life endangering if not acted upon in the right time and thus early diagnosis of pneumonia is vital.Chest X-rays are currently the best method for diagnosing pneumonia. X-ray images of pneumonia is not very clear and often misclassified to other diseases or other benign abnormalities. Moreover, the bacterial or viral pneumonia images are sometimes miss-classified by the experts, which leads to wrong medication to the patients and thereby worsening the condition of the patients.

**Challenges**

Detecting pneumonia in chest X-rays is a challenging task that relies on the availability of expert radiologists. There are considerable subjective inconsistencies in the decisions of radiologists were reported in diagnosing pneumonia. Detecting pneumonia in chest radiography can be difficult for radiologists. The appearance of pneumonia in X-ray images is often vague, can overlap with other diagnoses, and can mimic many other benign abnormalities. Therefore, there is a pressing need for a computer system to interpret chest radiographs as effectively as practicing radiologists which could thus provide substantial benefit in many clinical settings, from improved workflow prioritization and clinical decision support to large-scale screening and global population health initiatives.

**Proposed Solution**

Convolutional neural networks (CNNs) have shown great promise in image classification and therefore widely adopted by the research community. Deep Learning Machine learning techniques on chest X-Rays are getting popularity as they can be easily used with low-cost imaging techniques. 

Transfer learning can be useful in those applications of CNN where the dataset is not large. The concept of transfer learning is shown in the figure below, where the trained model from large dataset such as ImageNet can be used for application with comparatively smaller dataset.

![Transfer Learning](https://github.com/Mukundaram/Mukund/blob/master/images/Transfer_Learning.PNG)


## **Data Understanding**

**Dataset**

The dataset from kaggle chest X-ray pneumonia database was used and consists of chest X-ray images with resolution varying from 400 pixels to 2000 pixels. Chest X-ray images were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients’ routine clinical care. For the analysis of chest x-ray images, all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert.

Out of 5216 training images, there are 3875 images from different subjects affected by pneumonia (2530 images for bacterial pneumonia and 1345 images for viral pneumonia) and 1341 images are from normal subjects. Mixed viral and bacterial infection occurs in some cases of pneumonia. However, the dataset provided does not include any case of viral and bacterial co-infection.
The validation dataset consists of just 16 images (8 normal and 8 bacterial pneumonia images). Since this data was very small and does not contain instances of viral pneumonia, this was discarded and 10% of training data was used as validation dataset.

The test dataset consists of 390 pneumonia images (242 bacterial & 148 viral) and 234 normal chest X-ray images.

*Note: The focus is on classifying viral pneumonia and bacterial pneumonia and so, the normal chest X-ray images are not used for modelling purposes.*

# **Handling Class Imbalance using Weighted Loss Function**

The dataset is imbalanced as the number of bacterial pneumonia images is quite high than the number of virus pneumonia images. Class imbalance represents an important problem for intelligent classification algorithms.

The goal is to identify bacterial/viral pneumonia, but we don't have very many of those virus pneumonia samples (positive samples to work with) and so we would want to have the classifier heavily weight the few examples that are available. **This will cause the model to "pay more attention" to examples from an under-represented class.**

## **Data Preparation**

**Data labelling**

To start with, the bacterial pneumonia images are labelled as zero (negative examples) and the viral pneumonia images are labelled as one (positive examples). 10% of the training data is used for validation. 

***Note: There is no patient overlap between the training/validation and test sets.***

**Data Pre-processing**

Before inputting the images into the network, the following data pre-processing steps needs to be done.

**Step 1:** The images were resized to 224 pixels by 224 pixels – this depends on the pre-trained model that will be used for transfer learning. In this case, DenseNet121 was used and input shape must be (224, 224, 3). It should have exactly 3 inputs channels, and width and height should be no smaller than 32. E.g. (200, 200, 3) would be one valid value.
Note: The DenseNet121 model architecture will be explained in detail in the next section – Modelling.

**Step 2:** Normalize the data based on the mean and standard deviation (SD) of images in the ImageNet training set.

**Step 3:** For each image in the training set, the image is augmented with random horizontal flipping and zoom before being fed into the network. The augmentation is not applied for testing set.

All the above steps can be applied to the images dynamically using the ImageDataGenerator class and then flow_from_directory or flow_from_dataframe methods in Keras. The ImageDataGenerator class consists of lot of different options for augmenting the data. Refer to https://keras.io/api/preprocessing/image/#imagedatagenerator-class to know more. The below parameters were used in this case: 
 * <b>zoom_range:</b> 0.1 
 * <b>fill_mode:</b> ‘constant’ 
 * <b>horizontal_flip:</b> True 
 * <b>preprocessing_function:</b> preprocess_input 
 * <b>validation_split:</b> 0.1 (Use 10% of training data for validation) 

# **Predictive Modelling**

**Model Architecture**

The pre-trained model used for this problem is DenseNet which is widely used for image classification tasks using transfer learning and has yielded good results.

Dense Convolutional Network (DenseNet), connects each layer to every other layer in a feed-forward fashion. Whereas traditional convolutional networks with L layers have L connections - one between each layer and its subsequent layer - our network has L(L+1)/2 direct connections. For each layer, the feature-maps of all preceding layers are used as inputs, and its own feature-maps are used as inputs into all subsequent layers. DenseNets have several compelling advantages: they alleviate the vanishing-gradient problem, strengthen feature propagation, encourage feature reuse, and substantially reduce the number of parameters. DenseNets obtain significant improvements over the state-of-the-art on most of them, whilst requiring less memory and computation to achieve high performance.

![DenseNet121 Architecture](https://github.com/Mukundaram/Mukund/blob/master/images/DenseNet_Architecture.PNG)

## **Experimental setup**

![Model Architecture](https://github.com/Mukundaram/Mukund/blob/master/images/Model_Architecture.PNG)

The model used is a ***121-layer Dense Convolutional Network (DenseNet)*** trained on the ImageNet dataset. Use the pre-trained DenseNet121 model and add a ***Pooling layer, dropout layer*** and finally, a **dense layer with Sigmoid activation** for making the predictions.

**Step 1: Create the base model from the pre-trained convnets**

First, instantiate a DenseNet121 model pre-loaded with weights trained on ImageNet. In Keras, by specifying the include_top=False argument, we can load a network that doesn't include the classification layers at the top and add few customized layers based on the given task. 

**Step 2: Feature extraction** 

Freeze the convolutional base created from the previous step and to use as a feature extractor. Additionally, we will add classifier on top of it and train the top-level classifier.

**Freeze the convolutional base** 

It is important to freeze the convolutional base before you compile and train the model. In Keras, freezing (by setting layer.trainable = False) prevents the weights in each layer from being updated during training. DenseNet121 has many layers, so setting the entire model's trainable flag to False will freeze all of them. 

**Step 3: Add a classification head** 

To generate predictions from the block of features, the output from the final convolutional layer is passed through a pooling layer and dropout layer (for regularization). The final classification layer is replaced with a final fully connected layer with one that has a single output, after which we apply a sigmoid nonlinearity.

**Step 4: Build model**

Build a model by chaining together the data augmentation, image resizing, base_model (DenseNet121) and feature extractor layers using the Keras Functional API. Use training=False as our model contains a BatchNormalization layer.

**Step 5: Compile and fit the model**

Compile the model before training it. Since there are two classes, use a binary cross-entropy loss since the model provides a linear output. All parameters of the networks were trained jointly using Adam optimizer with standard parameters. Adam is an effective variant of an optimization algorithm called stochastic gradient descent, which iteratively applies updates to parameters in order to minimize the loss during training. The network was trained with minibatches of size 32. The initial learning rate was set as 0.0001 that was decayed by a factor of 10 each time the loss on the tuning set plateaued after an epoch (a full pass over the training set). In order to prevent the networks from overfitting, early stopping was performed, and the best model based on validation loss was saved after each epoch.

**Step 6: Train the model**

We train the model for few epochs initially. All the layers of the base model are not trainable and only the newly added classification layers are trained in this phase.

**Step 7: Fine tuning**

In the feature extraction step, model training was performed a few layers on top of a DenseNet121 base model. The weights of the pre-trained network were not updated during training. One way to increase performance even further is to train (or "fine-tune") the weights of the top layers of the pre-trained model alongside the training of the classifier you added. The training process will force the weights to be tuned from generic feature maps to features associated specifically with the dataset.

**Un-freeze the top layers of the model** 

All you need to do is unfreeze the base_model and train the various layers. Then, you should recompile the model (necessary for these changes to take effect), and resume training. This can be done is Keras by setting: base_model.trainable = True Note: Although the base model becomes trainable, it is still running in inference mode since we passed training=False when calling it when we built the model. This means that the batch normalization layers inside won't update their batch statistics. If they did, they would wreak havoc on the representations learned by the model so far.

**Compile the model**

As you are training a much larger model and want to readapt the pretrained weights, it is important to use a lower learning rate at this stage. We reduce the initial learning rate by a factor of 10.

**Continue training the model**

Train the model again for a few epochs and follow the same settings for learning rate reduction, early stopping and model checkpoint. This step will improve your accuracy by a good margin.

**Step 8: Evaluation and prediction**
Finally, we can verify the performance of the model on new data using test set. We can load the best model saved after every epoch based on validation loss and evaluate the performance on test set.

## **Model Evaluation**

**Model metrics**

The last step is to classify the attained data and assign it to a specific. The performance of the model for testing dataset is evaluated after the completion of training phase and was compared using six performances metrics such as accuracy, sensitivity or recall, Specificity, Precision, Area under curve (AUC) and F1 score.

**Area under curve (AUC)**

The Receiver Operator Characteristic (ROC) curve is an evaluation metric for binary classification problems. It is a probability curve that plots the True Positive Rate (TPR) against False Positive Rate (FPR) at various threshold values and essentially separates the ‘signal’ from the ‘noise’. The Area Under the Curve (AUC) is the measure of the ability of a classifier to distinguish between classes and is used as a summary of the ROC curve.


![Confusion Matrix](https://github.com/Mukundaram/Mukund/blob/master/images/Confusion_matrix.PNG)

![ROC Curve](https://github.com/Mukundaram/Mukund/blob/master/images/ROC.PNG)

# **Model Interpretability using Grad-CAM**

As a deep learning practitioner, one is responsible to ensure the model is performing correctly. One way you can do that is to debug your model and visually validate that it is “looking” and “activating” at the correct locations in an image.

To help deep learning practitioners debug their networks, Selvaraju et al. published a novel paper entitled, Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.
This method is:
* Easily implemented
* Works with nearly any Convolutional Neural Network architecture
* Can be used to visually debug where a network is looking in an image

***Gradient-weighted Class Activation Mapping (Grad-CAM)*** works by 
**(1) finding the final convolutional layer in the network and then (2) examining the gradient information flowing into that layer.**

***The output of Grad-CAM is a heatmap visualization for a given class label (either the top, predicted label or an arbitrary label we select for debugging). We can use this heatmap to visually verify where in the image the CNN is looking.***

For more information on how Grad-CAM works, refer to Selvaraju et al.’s paper - https://arxiv.org/abs/1610.02391 as well as this article by Divyanshu Mishra - https://towardsdatascience.com/demystifying-convolutional-neural-networks-using-gradcam-554a85dd4e48

![GradCam Virus](https://github.com/Mukundaram/Mukund/blob/master/images/Gradcams.PNG)

![GradCam Bacteria](https://github.com/Mukundaram/Mukund/blob/master/images/Gradcams2.PNG)

## **MODEL DEPLOYMENT**
The model weights have been saved in a hdf5 file. Hence, the model can be loaded any time and used to predict either on a single image or a batch of images.
1. Create the same model architecture and instantiate the weights of the model from the saved file.
2. Load the image and apply the data pre-processing steps.
3. Predict using the model and pre-processed image.
4. Use the GradCAM function to validate that it is “looking” and “activating” at the correct locations in an image.


## **Model Improvement**
Most of the research papers published in medical imaging domain train have used the below approaches to improve the model performance:
* Training on a lower Learning Rate in a bigger computing architecture may improve existing model metrics.
* Cross Validation using k-fold can further improve the model accuracy and AUC scores.
* Ensemble modelling by combining the predictions from multiple models.
* Collect/combine data from multiple sources to increase the dataset size.

## **Technical Stack**

**Programming language:** Python

**Deep Learning framework:** Keras and TensorFlow

**GPU backend :** Google Colab










