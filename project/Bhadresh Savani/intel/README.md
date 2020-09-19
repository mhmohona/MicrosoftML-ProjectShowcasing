# How to Download Model From OpenVino Zoo:

I have already downloaded four model in this repository from Intel Model Zoo. This model will work well for **Openvino Toolkit 2020.1v**
 
If you want to download pretrained models on your own here are the commands,

### Step1. Initialize Openvino Toolkit
```
cd C:\Program Files (x86)\IntelSWTools\openvino\bin\
setupvars.bat
```

### Step2. Download Models:
Note: in `--output_dir`, UserName indicates your computer User Name, You can also choose any path as `--output_dir`, Make sure to copy paste model in the Project Directory

1. gaze-estimation-adas-0002:
```
python "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\open_model_zoo\tools\downloader\downloader.py" --name gaze-estimation-adas-0002 --output_dir C:\Users\UserName\Documents\Intel\OpenVINO\openvino_models\models --cache_dir C:\Users\UserName\Documents\Intel\OpenVINO\openvino_models\cache
```
2. face-detection-adas-binary-0001:
```
python "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\open_model_zoo\tools\downloader\downloader.py" --name face-detection-adas-binary-0001 --output_dir C:\Users\UserName\Documents\Intel\OpenVINO\openvino_models\models --cache_dir C:\Users\UserName\Documents\Intel\OpenVINO\openvino_models\cache
```
3. head-pose-estimation-adas-0001:
```
python "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\open_model_zoo\tools\downloader\downloader.py" --name head-pose-estimation-adas-0001 --output_dir C:\Users\UserName\Documents\Intel\OpenVINO\openvino_models\models --cache_dir C:\Users\UserName\Documents\Intel\OpenVINO\openvino_models\cache
```
4. landmarks-regression-retail-0009:
```
python "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\open_model_zoo\tools\downloader\downloader.py" --name landmarks-regression-retail-0009 --output_dir C:\Users\UserName\Documents\Intel\OpenVINO\openvino_models\models --cache_dir C:\Users\UserName\Documents\Intel\OpenVINO\openvino_models\cache
```
