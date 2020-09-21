# import the necessary packages
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import numpy as np
import argparse
import imutils
import cv2
from glob import glob

import sys,os
sys.path.insert(1, "/content/drive/My Drive/Chest_X_rays/gradcam_function")

from compute_gradcam.gradcam import GradCAM #function to compute gradcam

def fn_gradcam(img_path):
  print("Image name: "+str(os.path.basename(os.path.normpath(img_path))))
  print("[INFO] loading model...")
  # initialize the model to be DenseNet121
  base_model = DenseNet121(weights=None, include_top=False)

  x = base_model.output

  # add a global average pooling layer
  x = GlobalAveragePooling2D()(x)

  #Dropout layer
  x = Dropout(0.2)(x) # Regularize with dropout
  
  # and a logistic layer
  predictions = Dense(1, activation="sigmoid")(x)

  model = Model(inputs=base_model.input, outputs=predictions)

  #Load the weights from hdf5 file
  model.load_weights("/content/drive/My Drive/Chest_X_rays/dn121_class_weights_best_model.hdf5")


  # load the original image from disk (in OpenCV format) and then
  # resize the image to its target dimensions
  orig = cv2.imread(img_path)
  resized = cv2.resize(orig, (224, 224))

  # load the input image from disk (in Keras/TensorFlow format) and
  # preprocess it
  image = load_img(img_path, target_size=(224, 224))
  image = img_to_array(image)
  image = np.expand_dims(image, axis=0)
  image = imagenet_utils.preprocess_input(image)

  # use the network to make predictions on the input imag and find
  # the class label index with the largest corresponding probability
  preds = model.predict(image)
  if np.round(preds) == 1:
    label = "Virus Pnenumonia"
    label = "{}: {:.3f}".format(label, preds[0][0])
  else:
    label = "Bacteria Pnenumonia"
    label = "{}: {:.3f}".format(label, preds[0][0])

  #decode the ImageNet predictions to obtain the human-readable label
  """decoded = imagenet_utils.decode_predictions(preds)
  (imagenetID, label, prob) = decoded[0][0]
  label = "{}: {:.2f}".format(label, prob * 100)
  print("[INFO] {}".format(label))"""
  #Above two lines not needed since this is a binary classification

  #label = "Virus Pneumonia"

  # initialize our gradient class activation map and build the heatmap
  cam = GradCAM(model, 0, 'bn') #We want the CAM for the batch normalization layer 
  heatmap = cam.compute_heatmap(image)

  # resize the resulting heatmap to the original input image dimensions
  # and then overlay heatmap on top of the image
  heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
  (heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5, colormap = cv2.COLORMAP_JET)

  # draw the predicted label on the output image
  cv2.rectangle(output, (0, 0), (600, 60), (0, 0, 0), -1)
  cv2.putText(output, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,1.2, (255, 255, 255), 2)

  # display the original image and resulting heatmap and output image
  # to our screen
  output = np.vstack([orig, heatmap, output])
  output = imutils.resize(output, height=700)
  from google.colab.patches import cv2_imshow
  #cv2.imshow("Output", output)
  cv2_imshow(output)
  cv2.waitKey(0)