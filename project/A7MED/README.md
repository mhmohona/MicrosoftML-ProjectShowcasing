# Deep learning model for COVID-19 diagnosis using chest X-ray images
Microsoft Azure Machine Learning Scholarship Project Showcasing Challenge

## Team:
@A7MED, @Jhonatan Tirado, @Radha Revathi G, @Lourdes Lizbeth Luna Rodríguez

## Guidelines: [https://github.com/mhmohona/MicrosoftML-ProjectShowcasing/blob/master/README.md](https://github.com/mhmohona/MicrosoftML-ProjectShowcasing/blob/master/README.md)

## Introduction
Nowadays, the world is suffering a health crisis, where it is essential to detect the problem in time and be able to do something about it. In December 2019, in Wuhan, China started an infectious disease called COVID-19, broadly speaking, this disease causes respiratory infections that can range from the common cold to more serious diseases that affect the lungs. 

It has been observed that around 80% of those who suffer from the disease recover without the need for hospital treatment, however, 1 in 5 people who contract COVID-19 end up presenting a serious condition and experiencing breathing difficulties, to such an extent that death can happen.

This disease has been alarming because of the speed with which it spreads, in addition to the complications that occur in the health of the person, as well as in the social context. An important factor to stop this disease, in addition to preventing it by taking hygiene and social distancing measures, for those who suffer from it, is early detection in order to take action in time. Therefore, in this project, a classifier for X-ray images of the rib cage is implemented on a web server, where the objective is to discriminate between healthy lungs, lungs with regular pneumonia or lungs with COVID-19 pneumonia.

As already mentioned, the project seeks to classify between three types of lung condition, since it has been seen that COVID-19 pneumonia is very similar to regular pneumonia, which has made it difficult to give diagnoses. Many investigations are being developed in order to differentiate between COVID-19 pneumonia and other types of pneumonia, since this type of information can be very useful for diagnoses and the simple understanding of how it is that COVID-19 affects the lungs. 

In order to be able to help the area of medicine, the implementation of an intelligent model available to everyone is innovative, practical and useful to provide the appropriate service to the patient. 

For the project, the steps to follow were: 
   1) Obtaining, preparing and exploring the data.
   2) Design, training and validation of a deep neural network.
   3) Testing of the deep neural network. 
   4) Deployment for real time inferencing.
   5) Evaluation of the implementation in a web server.

In addition, it is also important to mention that for the development and simulation of this project we used the **PyTorch** library.

## Evaluation criteria:
1. Using Azure for Implementation based on Course Material (30%) =
   1. The labs in the course did not show how to train or deploy a model for image classification.
   2. We deployed the model, as a REST API endpoint for real time inferencing, to a virtual machine in the cloud.
2. Innovation & Creativity (20%) = Evaluation of the novelty, innovation and creativity introduced in the project such that it is appealing.
   1. The model can predict if an X-ray image shows normal lungs, lungs infected with common pneumonia, or lungs infected with COVID-19. This is important to make a difference between common pneumonia and the one caused by COVID-19.
3. Project Implementation (20%) = Evaluation of how much the planned idea was implemented in this project and how well the results are presented.
   1. To improve accuracy of the model, we planned to train 3 binary classifiers and use ensemble learning to determine the final result by a voting mechanism
      * First model would have two classes: "normal" and "not normal"
      * Second model would have two classes: "pneumonia" and "not pneumonia"
      * Third model would have two classes: "covid-19" and "not covid-19"
   2. Thus, when we want to predict if an X-ray is normal, pneumonia or covid-19, we would feed the same image to the three different models. Then, we would use a voting mechanism to determine the final result.
   3. For instance, if the input image is "covid-19", the expected results are:
      * First model: "not normal"
      * Second model: "not pneumonia"
      * Third model: "covid-19"
4. Impact & Potential (15%) =
   1. The model could be a tool to aid physicians to diagnose COVID-19. There’s a shortage of PCR testing kits, the gold standard to diagnose COVID-19, in countries like Peru, where a blood test to detect coronavirus antibodies is used instead.
5. Responsible AI (15%) = Evaluation of the potentiality of the project which is fair, inclusive to everyone, preserves data privacy and is secure.
   1. The model preserves data privacy because no PII (Personal Identifiable Information) is used. The input is a chest X-ray, and the output is a label (normal, pneumonia or COVID-19) and a probability (0 to 1).
   2. The model could be secured if access restrictions are added to the endpoint, to prevent unauthorized use
   3. The model is as inclusive as its dataset. If most of the chest X-ray images belong to asian patients, then the model can better predict COVID-19 for asian patients than for african people. That’s why it is important to collect data from all races, age and gender.

## Dataset
Chest radiography images were used in this project. Two different sub-databases were used to create one database.

Images of healthy lungs and regular pneumonia were acquired from Kaggle databases which are publicly available and it’s found in the next link: [https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia?](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia?)

While lungs with COVID-19 pneumonia were extracted from a publicly available GitHub database, which is found on the below link: 
[https://github.com/lindawangg/COVID-Net/blob/master/docs/COVIDx.md](https://github.com/lindawangg/COVID-Net/blob/master/docs/COVIDx.md)

The images used in this project of the three types of lungs were used in .jpeg format, therefore it is considered as a requirement of the application.

The final dataset used had a total of 3093 images of X-ray, which have been divided into three groups and each one has three classes of X-ray images that belong to normal lungs, regular pneumonia lungs and COVID-19 pneumonia lungs. Below, the number of images used for each group is showed on Fig. 1.

**Figure 1.** Number of X-ray images used in the deep learning model for COVID-19 diagnosis.  
![Table](https://github.com/waqasne/MicrosoftML-ProjectShowcasing/blob/master/project/A7MED/images/number_images.png)

## Data Preprocessing
The X-ray images were resized to 224 x 224 and converted to type *Tensor*. The following code is used to do the mentioned preprocessing:

   *data_transform = transforms.Compose\(\[transforms.Resize(size=(224,224)),transforms.ToTensor()\]\)*

## Model Architecture
A Convolutional Neural Network (CNN) was implemented for this project, in the Fig. 2 is show the final structure of this CNN, we can note that this CNN has 10 layers, where the Conv2d layers uses the function ReLu like its activation function.

**Figure 2.** Structure of the CNN.   
![model](https://github.com/waqasne/MicrosoftML-ProjectShowcasing/blob/master/project/A7MED/images/model_CNN.png)

## Model Training and Validation
Hyperparameters:  
      A. Learning rate = 0.05  
      B. Training epochs = 1  
      C. Batch size = 20  
      
Optimizer:  
      A. Stochastic Gradient Descent (SGD)  
      
Loss function:  
      A. Cross Entropy  

## Model Testing
Accuracy for COVID-19 Chest X-Rays = 95.0%, 175/184  
Accuracy for NORMAL Chest X-Rays = 76.0%, 276/362  
Accuracy for PNEUMONIA Chest X-Rays = 94.0%, 368/390  

**Test Accuracy**: 87% (819/936)  

## Deployment for Real Time Inferencing 

The code is available on GitHub at: [https://github.com/jhonatantirado/covid-azure-ml/blob/master/serve_rest_api.py](https://github.com/jhonatantirado/covid-azure-ml/blob/master/serve_rest_api.py)

The deep learning classification model was trained and generated by pytorch as a ".pt" file, which includes the calculated weights and biases of the model, but not the structure.

The trained model was then deployed as a REST API endpoint that can be consumed by a mobile or web application.

It’s important to load the model only once on startup to avoid loading the model everytime a request is sent to the REST API endpoint and to curb memory consumption.

**Figure 3.** Code snippet where the pytorch model is loaded as a global variable.

![Load model](https://github.com/waqasne/MicrosoftML-ProjectShowcasing/blob/master/project/A7MED/images/LoadModel-CodeSnippet.png)

### Python libraries used

   1. flask: allows to run a REST API endpoint using Python
   2. flask_cors: allows to set access permissions on the REST API, for security reasons
   3. urllib.request: to download the image file and save it to the local computer or server
   4. PIL: to open the image file and convert it to RGB
   5. torchvision.transforms: to resize the image and convert it to the pytorch’s tensor format as requested by the convolutional neural network
   6. torch: to load the model
   7. torch.nn: to calculate the Softmax probability distribution of the output
   8. flask.jsonify: to return the output in JSON format

**Figure 4.** Code snippet where the REST API endpoint is defined.

![Rest API defined](https://github.com/waqasne/MicrosoftML-ProjectShowcasing/blob/master/project/A7MED/images/RESTAPIEndpoint-CodeSnippet.png)

### Deployment

The model was deployed to a virtual machine (VM) in the cloud, but the VM is shutdown to avoid costs ($). It could be deployed to the Microsoft Azure cloud, or any other cloud provider.

It was started from the command line

   *  Endpoint: /todo/api/v1.0/covid19
   
**Figure 5.** REST API endpoint running in the command line.

![Rest API running](https://github.com/waqasne/MicrosoftML-ProjectShowcasing/blob/master/project/A7MED/images/FlaskEndpointRunning.png)

### Testing the implementation

You can use POSTMAN or any similar HTTP client to send requests to the endpoint

### Input: POST request

The following is a sample expected input in JSON format.

{
"imageUrl": "https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQEAwwYsXILZIrDXx7MYIqECR2uTosvETA3qQ&usqp=CAU"
}

*imageUrl:* is the link to the image. The chest X- ray image in JPEG format should be uploaded to any public website. The image URL should be sent as input in the request, because sending a string is easier and quicker than sending an actual image file

**Figure 6.** POST request in JSON format using the HTTP client POSTMAN.

![POST_request](https://github.com/waqasne/MicrosoftML-ProjectShowcasing/blob/master/project/A7MED/images/POSTRequestJSONPostman.png)

### Output: POST response

The REST API endpoint receives the request, processes it and returns a response in JSON format.

*label:* is the predicted class for the chest X-ray image submitted in the request
probability: the probability of the image being the class indicated in the label


**Figure 7.** POST response in JSON format using the HTTP client POSTMAN.

![POST_response](https://github.com/waqasne/MicrosoftML-ProjectShowcasing/blob/master/project/A7MED/images/POSTResponseJSONPostman.png)

## Conclusions  
In this project, we created a CNN model that classifies three types of chest X-ray images, where we can find normal lungs, normal pneumonia lungs, and COVID-19 pneumonia lungs. This model showed an accuracy of 87% for the test set.

Also, as a first test, this model was implemented on a local server. This with the aim of being able to help detect the COVID-19 disease in an easy and fast way.
