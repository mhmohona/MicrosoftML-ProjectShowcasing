# 👀[Eye Excercise App](https://github.com/bhadreshpsavani/EyeExerciseApp)👀
The goal of the Eye Exercise Application is to help increase effectiveness of eye exercise. We detect eye gaze of a user using computer vision and pretrained machine learning models. We check it with eye gaze coordianates of instructor and show user a live score of exercise effectiveness. We also want the exercise assistant to be voice enabled to instruct and motivate users for exercise.

## How we build eye gaze vectors:
![demoVideo](/project/Bhadresh%20Savani/bin/gaze_detection.gif)

We have used four pretrained machine learning models from OpenVINO™ toolkit, short for Open Visual Inference and Neural network Optimization toolkit:

1. [Face Detection](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html): Detects face coordinates from video or webcam images
2. [Head Pose Estimation](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html): Detects pose coordinates for head
3. [Facial Landmarks Detection](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html): Gives coordinates or location for facial landmarks like eyes, nose and mouth
4. [Gaze Detection Model](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html): Takes head pose coordinates and eye landmark as input and predicts gaze vector

### 🏛️ The Pipeline:
![pipeline](/project/Bhadresh%20Savani/imgs/pipeline.png)

## How we get Exercise Score:
We compare eye gaze vector of instructor and User using cosine similarity. 
```
>> from scipy.spatial.distance import cosine
>> eye_gaze_instructor = [ 0.62916321,  0.10232677, -0.77875257]
>> eye_gaze_user = [ 0.09647849,  0.03398839, -0.82852501]
>> cosine(eye_gaze_instructor, eye_gaze_user)
0.15561332345537238
```

## Project Set Up and Installation:

Step 1. Download below three tools:
1. Microsoft Visual Studio with C++ 2019, 2017, or 2015 with MSBuild
*If you want to use Microsoft Visual Studio 2019, you are required to install CMake 3.14.
2. CMake 3.4 or higher 64-bit
3. Python 3.6.5 64-bit

Step 2. Download **[OpenVINO™ toolkit 2020.1](https://docs.openvinotoolkit.org/latest/index.html)** with all the prerequisites by following this [installation guide](https://docs.openvinotoolkit.org/2020.1/_docs_install_guides_installing_openvino_windows.html)

Step 3. Setup OpenVINO™ toolkit 2020.1 using the commands below in command prompt
```
cd C:\Program Files (x86)\IntelSWTools\openvino\bin\
setupvars.bat
```

Step 4. Configure model optimizer using the commands below in command prompt
```
cd C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\model_optimizer\install_prerequisites
install_prerequisites.bat
```

Step 5. Verify installation
```
cd C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\demo\
demo_squeezenet_download_convert_run.bat
```
Above commands should give output like this image
![optimizer_output](/project/Bhadresh%20Savani/imgs/image_classification_script_output_win.png)

## 🔎[Demo]((/bin/EyeExcerciseDemoVideo.mp4)):
![demoVideo](/project/Bhadresh%20Savani/bin/EyeExcerciseDemoVideo.gif)

### Instructions: 
Step 1. Clone the repository using `git clone https://github.com/bhadreshpsavani/EyeExerciseApp.git`

Step 2. Create virtual environment using command `python -m venv base` in the command prompt, then activate environment using the commands below,
```
cd base/Scripts/
activate
```

Step 3. Install all the dependency using `pip install requirements.txt`.

Step 4. Instantiate OpenVINO™ toolkit environment. For Microsoft Windows use the commands below
```
cd C:\Program Files (x86)\IntelSWTools\openvino\bin\
setupvars.bat
```

Step 5. Go back to the project directory `src` folder
```
cd path_of_project_directory
cd src
```

Step 6. Run the commands below to execute the project
```
python main.py -fd ../intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml -lr ../intel/landmarks-regression-retail-0009/FP32-INT8/landmarks-regression-retail-0009.xml -hp ../intel/head-pose-estimation-adas-0001/FP32-INT8/head-pose-estimation-adas-0001.xml -ge ../intel/gaze-estimation-adas-0002/FP32-INT8/gaze-estimation-adas-0002.xml -i cam
```
Command line argument information:
- fd : Specify path of xml file of face detection model
- lr : Specify path of xml file of landmark regression model
- hp : Specify path of xml file of heead pose estimation model
- ge : Specify path of xml file of gaze estimation model
- i : cam for Webcam

### Output Video:
![demoVideo](/project/Bhadresh%20Savani/bin/output.gif)


## 📚Documentation: 

### Project Structure:

![project_structure](/project/Bhadresh%20Savani/imgs/project_structure.png)

**bin**: This folder has `demo.mp4` file which we are using for eye excercise video

**imgs**: It contains images used in this project for documentations and results

**intel**: This folder contains machine learning models in IR(Intermediate Representation) format

**src**: This folder contains model files, pipeline file(main.py) and utilities 
* `model.py` is the model class file which has common property of all the other model files. It is inherited by all the other model files 
This folder has 4 model class files that has methods to load model and perform inference.
  * `face_detection_model.py`
  * `gaze_estimation_model.py`
  * `landmark_detection_model.py`
  * `head_pose_estimation_model.py`
* `main.py` file used to run complete pipeline of project. It has object of all the other class files in the folder
* `input_feeder.py` is utility to load local video or webcam feed

## Technology:
* Computer Vision
* Openvino Toolkit
* Python
* Microsoft Azure for Deployment(Future Step)

## Challenges:
* Reduce lag between Instructor and User Video
* Handle Different different lighting conditions
* Increase Inference Speed

## ✨Team:
* [Bhadresh Savani](https://github.com/bhadreshpsavani)
* [Pakeeza](https://github.com/Hotaru29)
* [Erin Song](https://github.com/sagabanana)
* [Richa](https://www.linkedin.com/in/richaphd/)
* [Jose Mariscal](https://github.com/jgmarsm) 

## 🧱Road Map:
- [x] Create end to end pipeline to extract eye gaze coordinates
- [x] Create pipeline for getting eye gaze coordinates for excersice video
- [x] Create pipeline and UI for webcam video
- [x] Develop score computation logic
- [x] Develop UI for computer to view output and score
- [ ] Enable app with voice assistance
- [ ] Deploy application to Microsoft Azure Cloud
- [ ] Create a mobile app
