# Check TF version
import tensorflow as tf
print(tf.__version__)

import os
import sys
sys.path.append("MONK/Monk_Object_Detection/13_tf_obj_2/lib/")

from train_detector import Detector

gtf = Detector();

print(gtf.list_models())


train_img_dir = "results/Train/images";
train_anno_dir = "results/Train/annotations";
class_list_file = "pascal-voc-classes.txt";

gtf.set_train_dataset(train_img_dir, train_anno_dir, class_list_file, batch_size=24, trainval_split = 0.8)

## Output dir
output_dir = os.path.join("data_tfrecord")

gtf.create_tfrecord(data_output_dir=output_dir)
