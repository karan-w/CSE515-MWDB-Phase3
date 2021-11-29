from os import listdir
from os.path import isfile, join
import numpy  as np
import os
import shutil

# if __name__=="main":
org_path = "all/"

images = [f for f in listdir(org_path) if isfile(join(org_path, f))]
print(len(images))
image_dict = dict()
for image in images:
    tt=image.split("-")[3]
    if tt in image_dict:
        image_dict[image.split("-")[3]].append(image)
    else:
        image_dict[image.split("-")[3]] = [image]

train_keys = list(image_dict.keys())[:6]
test_keys = list(image_dict.keys())[5:]
train_set=[]
test_set=[]
for k in train_keys:
    for te_img in image_dict[k]:
        train_set.append(te_img)
for k in test_keys:
    for te_img in image_dict[k]:
        test_set.append(te_img)
print("hi")
# os.mkdir("test")
# os.mkdir("train")
for test_img in test_set:
    src_full_file_name = os.path.join("all/", test_img)
    # dst_full_file_name = os.path.join("test/", test_img)
    shutil.copy(src_full_file_name, 'test/'+test_img)

for train_img in train_set:
    src_full_file_name = os.path.join("all/", train_img)
    # dst_full_file_name = os.path.join("train/", train_img)
    shutil.copy(src_full_file_name, "train/"+train_img)



