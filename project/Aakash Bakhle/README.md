# Car make, model and year detection

## Udacity's Microsoft Azure Scholarship Project Showcasing Challenge:

Upload a car photo, identify it's model, make and the year in which it was manufactured. <br></br>
![cmm1](https://user-images.githubusercontent.com/48802744/91646051-efdda480-ea68-11ea-992e-7b5105f202d4.gif)
<i> Images uploaded are neither stored nor re-used for model training. </i>

## Dataset
Car make and model detection using Stanford dataset which was cleaned up and put on kaggle: <br></br>
https://www.kaggle.com/jutrera/stanford-car-dataset-by-classes-folder

The dataset consists of 16k+ images and has 196 distinct labels.<br></br>
Architecture used: ResNext50_32x4d using fastai API

## Author: 
Aakash Bakhle

## App trial:
Google one of the cars from the car_names in sidebar. Download any image and upload for model to predict. <br></br>
https://caridentifier.azurewebsites.net/


P.S: Link maynot be available always as free tier does not support systems with higher RAM.

## Running app on Microsoft Azure App service:

This service is not free and works only on S3 or P1v2 and above instances. Pricing can be found [here](https://azure.microsoft.com/en-gb/pricing/details/app-service/windows/?WT.mc_id=azureportalcard_Service_App%20Services_-inproduct-azureportal)

You need a free-tier/paid subscription with Microsoft Azure at https://portal.azure.com/ <br></br>

1. Search/look for 'App Services'
2. Click 'Create App Service'
3. Fill details. Create a new resource group with any name. <br></br>
   Select Docker Container (Region, Linux Plan and SKU Size come by default) and click next :
![image](https://user-images.githubusercontent.com/48802744/91644360-e9473100-ea58-11ea-8a75-ea3e35a7c32c.png)

4. From 'Image Source' dropdown, choose Docker Hub. After filling everything, click 'Review+create':
![image](https://user-images.githubusercontent.com/48802744/91644385-26abbe80-ea59-11ea-90a3-d45f46d48765.png)

5. Click 'Create'. It takes about 5 minutes. Once done click 'Go to Resource'.

6. Select 'Configuration' in the 'Settings' tab on the left side pane.:
![image](https://user-images.githubusercontent.com/48802744/91644581-02e97800-ea5b-11ea-95c3-32118822bdcf.png)

7. Click 'New Application Setting'
![image](https://user-images.githubusercontent.com/48802744/91644642-9ae76180-ea5b-11ea-84ba-30fc248baf39.png)

8. Fill in data as given below and click OK:
![image](https://user-images.githubusercontent.com/48802744/91644667-e1d55700-ea5b-11ea-8d1c-a42f98419678.png)

9. Click 'Save' and then Click 'Continue' for the Service to restart:
![image](https://user-images.githubusercontent.com/48802744/91644683-134e2280-ea5c-11ea-8754-ffb428efef52.png)

10. Head to the 'Overview' tab in the side pane and launch the URL:
![image](https://user-images.githubusercontent.com/48802744/91644737-8e173d80-ea5c-11ea-8357-6851ebc8e699.png)

11. Google one of the cars from the car_names in sidebar. Download any image and upload for model to predict. 
![image](https://user-images.githubusercontent.com/48802744/91644803-3b8a5100-ea5d-11ea-9010-1ba1ccb34f84.png)

12. Play around!
![image](https://user-images.githubusercontent.com/48802744/91644785-17c70b00-ea5d-11ea-820c-b69a18c4d2b1.png)

## Running the app on local system:

I assume you have miniconda/anaconda installed and your system has at least 2GB RAM.

Due to GitHub's upload limits, the model file can be found [here](https://drive.google.com/file/d/1eDUeSsMCt5aTHJ291766mFwqlMjYtyPb/view?usp=sharing) <br></br>
Clone this repo, place the model file in the same folder.

### Install dependencies
```bash
conda create --name demo
conda install pip
pip install -r requirements.txt
```
### Running the app
```bash
streamlit run app.py
```
Streamlit outputs localhost url. Open it and follow steps 11 and 12 from Azure Deployment above.

# Known issues:
1. Since source dataset mostly has images of front and side face of cars, model struggles to identify images taken of car's rear.
2. Model gets confused between cars from the same manufacturers.

# Further scope of improvement
1. More images of different cars captured from different angles. 
2. Multi label, Multi car data  in a single image, in low resolution for a generic use case.
3. More data augmentation and deeper understanding of working of CNNs.
4. Better usage of pretrained models.


# Files
  stanford_car_model_dataset_fastai.ipynb : 
  Contains the main code to train the model. <i><b>It is specific to google colab </b></i>
  
  app.py:
  Contains streamlit code to deploy model on localhost.
  
  labels.csv:
  List of car names from the original dataset
  
  Dockerfile:
  The dockerfile given has been built and pushed to [dockerhub](https://hub.docker.com/r/aakashbakhle/streamlit)
 
