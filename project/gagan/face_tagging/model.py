import os
from typing import Dict, List, NoReturn

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneOut
from tqdm import tqdm

from face_tagging.get_keypoints import get_keypts
from face_tagging.visualization import visualize


class FaceTaggingModel:
    def __init__(self, keypoints_df: pd.DataFrame=None, train_images_path: str=None, use_docker: bool=False):

        if not use_docker:
            if keypoints_df is None:
                raise ValueError("You have to use dataframe is use_docker = False")
            else:
                if 'Unnamed: 0' in keypoints_df.columns:
                    keypoints_df = keypoints_df.set_index('Unnamed: 0')

        self._df = keypoints_df
        self._silent = False
        self._groupedClasses = None
        self._user_docker = use_docker
        self._images_to_keypoints = dict()
        self._train_images_path = train_images_path
    
    def _prepare_data(self):
        images = os.listdir(self._train_images_path)
        if images == []:
            raise AttributeError("train_images folder is empty")

        for image in images:
            if image.split('.')[-1] not in ['jpg', 'png', 'jpeg']:
                continue
            url = os.path.join(self._train_images_path, image)
            result = get_keypts(url, self._silent)
            if result is not None:
                self._images_to_keypoints[image] = result
        self._df = pd.DataFrame.from_dict(self._images_to_keypoints).transpose()
    
    def _get_dataframe(self) -> pd.DataFrame:
        if self._user_docker:
            self._prepare_data()
        return self._df

    def train_model(self, silent: bool = False) -> Dict:
        self._silent = silent
        df = self._get_dataframe()
        kf_loo = LeaveOneOut()

        knn = KNeighborsClassifier(n_neighbors=1, algorithm='ball_tree', weights='distance')
        le = LabelEncoder()

        # dict to store the grouped clusters, classNum is the cluster number
        # groupedClasses will store the groups in the format:
        # {
        #   0: [image_name_1, image_name_2, image_name_3],
        #   1: [image_name_4],
        #   2: [image_name_5, image_name_6], ...
        # }  where 0, 1, 2, etc are the different class numbers

        groupedClasses = dict()
        classNum = 0

        # a set to store names of already predicted and matched images, so that we don't have to train on it again
        matchedImages = set()

        new_labels = np.ones(len(df), dtype=int) * -1

        # loop over the data to train and predict
        for predict_indices, train_index in tqdm(kf_loo.split(df)):
            train_index = train_index[0]
            
            label = df.iloc[train_index].name
            
            # if the label (image) was already predicted and matched before continue without training
            if label in matchedImages:
                continue
                
            label_encoded = le.fit_transform([label])
            
            # fit the KNN model on the single image
            points = df.iloc[train_index]
            knn.fit([points], label_encoded)
            
            # predict using the trained model on remaining indices to get the distances of each point to the trained image
            prediction = knn.kneighbors(df.iloc[predict_indices])    
            distances = prediction[0].flatten()
            
            # create a boolean array to filter out distances <= 0.5
            distanceFilter = distances <= 0.5
            
            # filter out the images and distances using the distance filter
            prediction_labels = np.array(df.iloc[predict_indices].index)
            similarLabels = prediction_labels[distanceFilter]
            similarDistances = distances[distanceFilter]

            # group the trained label and predicted similar labels into one class
            groupedClasses[classNum] = np.array([label])
            new_labels[train_index] = classNum
            
            if len(similarLabels) > 0:        
                groupedClasses[classNum] = np.append(groupedClasses[classNum],similarLabels)
                new_labels[predict_indices] = classNum
                if not silent:
                    print(f"\nKNN Image = {label}")
                matchedImages.add(label)
            classNum += 1
                
            for i in range(len(similarLabels)):
                if not silent:
                    print(f"Prediction Image = {similarLabels[i]}, Distance(s) = {similarDistances[i]}")
                # Keep track of predicted similar images so as to not train on them again
                matchedImages.add(similarLabels[i])
        self._groupedClasses = groupedClasses

        # knn_full = KNeighborsClassifier(n_neighbors=classNum, algorithm='ball_tree', weights='distance')
        # knn_full.fit(df, new_labels)
        
        print("Trained")

    def matched_groups(self) -> Dict:
        if self._groupedClasses is None:
            raise AttributeError("Model not trained")
        return self._groupedClasses


if __name__ == "__main__":
    obj = FaceTagging()
    obj.train_model()
    print(obj.matched_groups())
