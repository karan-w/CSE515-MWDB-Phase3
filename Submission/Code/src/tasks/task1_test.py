import random
from os import listdir
from os.path import isfile, join
import numpy  as np
import argparse
import logging
import time
from task_helper import TaskHelper
from utils.image_reader import ImageReader
from scipy.spatial import distance
import sys
import os
import pandas as pd
from utils.output import Output

import networkx

# logger = logging.getLogger(Task1.__name__)
logging.basicConfig(filename="logs/logs.log", filemode="w", level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S')

# if __name__=="main":
mypath="all/"

# Read all images

start_time = time.time()
image_reader = ImageReader()

train_path="E:\\projects\\workspace\\1000\\1000"
# train_path="all/"
# test_path="100/100/"

test_path="E:\\projects\\workspace\\100\\100"

train_images = image_reader.get_all_images_in_folder(train_path)
train_images_names = image_reader.get_all_images_filenames_in_folder(train_path)


print("train images name ............. ",len(train_images_names))

train_all_labels = [label.split("-")[1] for label in train_images_names]


print("train label list............. ",len(train_all_labels))
label_list = set()
label_list.update(train_all_labels)
print("label list............. ",len(label_list))


test_files = os.listdir(test_path)

for file in test_files:
    if "test" not in file and "image-" in file:
        splts = file.split(".")
        os.rename(os.path.join(test_path,file),os.path.join(test_path,splts[0]+"-test"+"."+splts[1]))


test_images = image_reader.get_all_query_images_in_folder(test_path)
test_images_names = image_reader.get_all_images_filenames_in_query_folder(test_path)

test_all_labels = [label.split("-")[1] for label in test_images_names]


print(len(test_images))

print(len(test_images[2].matrix))
# combined_images=[]
# combined_image_names=[]

test_labels_for_train = ["test"]*len(test_all_labels)

combined_images = [*train_images,*test_images]

combined_image_names = [*train_images_names,*test_images_names]

# combined_image_names2 = [*train_images_names,*test_labels_for_train]
combined_labels=[*train_all_labels,*test_all_labels]

print(len(combined_images))
print(combined_images[2].matrix)

feature_model="HOG"
dimensionality_reduction_technique="PCA"
k=20

task_helper = TaskHelper()
combined_images = task_helper.compute_feature_vectors(
    feature_model,
    combined_images)

print("Features calculated")
combined_images, drt_attributes = task_helper.reduce_dimensions(
    dimensionality_reduction_technique,
    combined_images,
    k)

print("Dimensions reduced")

rdfv = drt_attributes['reduced_dataset_feature_vector']

print("hi")
print(type(rdfv))
print(len(rdfv))
image_feature_map = compute_image_feature_map(combined_image_names,rdfv)

print("calculated image feature map of length ",len(image_feature_map))

print("combined length ",len(combined_image_names))

label_images_map = calculate_label_images_map(train_images_names,train_all_labels)

# label_images_map = calculate_label_images_map(combined_image_names,combined_labels)
print("mapped label to images")

# time.time()
print("---- %s seconds " % (time.time() - start_time))

random.shuffle(combined_images)

X=[]
y = [0] * len(combined_images)






