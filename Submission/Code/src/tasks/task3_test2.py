from os import listdir
from os.path import isfile, join
import numpy  as np
import argparse
import logging
import time
from task_helper import TaskHelper
from utils.image_reader import ImageReader
from scipy.spatial import distance

import pandas as pd
from utils.output import Output

def calculate_efficiency(image_label_dict, test_all_labels):
    count=0
    for i,j in zip(image_label_dict.values(),test_all_labels):
        print(i, j)
        if i==j:
            count+=1
    print("correct "+count)
    rate = count/len(test_all_labels)
    print("eff. "+rate)

def ppr(teleportation_matrix, random_walk, alpha):
    identity_matrix = np.identity(len(random_walk),dtype=float)
    inv = np.linalg.inv(identity_matrix - random_walk)
    pi = np.dot(inv,teleportation_matrix)
    return pi

def compute_image_feature_map(image_list,feature):
    img_feature_map=dict()
    for i,j in zip(image_list,feature):
        img_feature_map[i]=j.real.tolist()
    return img_feature_map

def calculate_label_images_map(train_images_names, train_all_labels):

    label_images_map = dict()
    for i in range(len(train_all_labels)):
        if train_all_labels[i] in label_images_map:
            label_images_map[train_all_labels[i]].append(train_images_names[i])
        else:
            label_images_map[train_all_labels[i]] = [train_images_names[i]]
    return label_images_map


def compute_seed_matrix(label_images_list,alpha=0.85):

    teleportation_matrix = [[0.0 for j in range(1)] for i in range(len(label_images_list))]

    for i in range(len(label_images_list)):
        teleportation_matrix[i][0]=(1-alpha)/len(label_images_list)
    teleportation_matrix = np.array(teleportation_matrix)
    return teleportation_matrix

# def compute_random_walk(image_feature_map,alpha=0.85,kk=10):
#     similarity_matrix=dict()
#
#     # random_walk = [[0.0 for i in range(len(image_feature_map.keys()))] for j in range(len(image_feature_map.keys()))]
#     random_walk=[]
#
#     for image1, feature1 in image_feature_map.items():
#         similar_list=[]
#         for image2, feature2 in image_feature_map.items():
#             if image1!=image2:
#                 # dist = distance.cityblock(feature1,feature2)
#                 similar_list.append(distance.cityblock(feature1,feature2))
#             else:
#                 # dist=0
#                 similar_list.append(0)
#         similar_list_truncated = sorted(similar_list, reverse=True)[:kk]
#         for i in range(len(similar_list)):
#             if similar_list[i] in similar_list_truncated:
#                 similar_list_truncated.remove(similar_list[i])
#             else:
#                 similar_list[i]=0
#
#         similar_list = [(alpha*i)/sum(similar_list) for i in similar_list]
#         random_walk.append(similar_list)
#     random_walk = np.array(list(map(list,zip(*random_walk))))
#
#
#
#
#
#
#
#
#
#     # for image1, feature1 in image_feature_map.items():
#     #     for image2, feature2 in image_feature_map.items():
#     #         if image2 != image1:
#     #             # print('Image1', image1, 'feature1', feature1)
#     #             # print('Image2', image2, 'feature2', feature2)
#     #             dist = distance.cityblock(feature1, feature2)
#     #             # print(image1,image2,dist)
#     #             if image1 in similarity_matrix:
#     #                 similarity_matrix[image1].append(tuple((image2, dist)))
#     #             else:
#     #                 similarity_matrix[image1] = [tuple((image2, dist))]
#     # for image, similarity_list in similarity_matrix.items():
#     #     similarity_matrix[image] = sorted(similarity_matrix[image],
#     #                                                     key=lambda x: x[1])[: kk]
#     #
#     # similarity_matrix = list(map(list, zip(*similarity_matrix)))
#     # similarity_matrix = np.array(similarity_matrix)
#     return random_walk

def similarity_of_a_image(image1,feature1):
    similar_list=[]
    for image2, feature2 in image_feature_map.items():
        if image1 != image2:
            # dist = distance.cityblock(feature1,feature2)
            similar_list.append(distance.cityblock(feature1, feature2))
        else:
            # dist=0
            similar_list.append(0)
    return similar_list


def compute_random_walk(image_feature_map,alpha=0.85,kk=20):
    # random_walk = [[0.0 for i in range(len(image_feature_map))] for j in range(len(image_feature_map))]
    # random_walk = random_walk.to(device)
    random_walk=list()
    for image1, feature1 in image_feature_map.items():
        print("img : ",image1)
        similar_list=[]

        # ========================================================

        for image2, feature2 in image_feature_map.items():
            if image1!=image2:
                # dist = distance.cityblock(feature1,feature2)
                similar_list.append(distance.cityblock(feature1,feature2))
            else:
                # dist=0
                similar_list.append(0)

        # ----------------------------------------------------------


        # ==========================================================


        similar_list_truncated = sorted(similar_list, reverse=True)[:kk]
        for i in range(len(similar_list)):
            if similar_list[i] in similar_list_truncated:
                similar_list_truncated.remove(similar_list[i])
            else:
                similar_list[i]=0
        # similar_list = [(alpha*i)/sum(similar_list) for i in similar_list]
        summ=sum(similar_list)
        similar_list = [(alpha*i)/summ for i in similar_list]
        random_walk.append(similar_list)

    print("---- %s seconds " % (time.time() - start_time))

    # random_walk = np.array(list(map(list,zip(*random_walk))))
    return random_walk

def associate_labels_to_test_images(test_images_names,labelled_ppr,label_list):

    label_count=dict()

    image_label_dict=dict()

    for l in label_list:
        label_count[l]=-1

        for i in range(len(test_images_names)):
            for j in labelled_ppr.keys():
                for k in range(len(labelled_ppr[j])):
                    if labelled_ppr[j][0] == l:
                        label_count[l]=k
                        break
        mini = 10000
        labell=""
        for lbl,indx in label_count.items():
            if mini>indx:
                mini=indx
                labell = lbl

        image_label_dict[l]=labell

    return image_label_dict


# def compute_similarity_matrix():

class Task1:
    def __init__(self):
        parser = self.setup_args_parser()
        # input_images_folder_path, feature_model, dimensionality_reduction_technique, reduced_dimensions_count, classification_images_folder_path, classifier
        self.args = parser.parse_args()


    def setup_args_parser(self):
        parser = argparse.ArgumentParser()

        # parser.add_argument('--input_images_folder_path', type=str, required=True)
        # parser.add_argument('--feature_model', type=str, required=True)
        # parser.add_argument('--dimensionality_reduction_technique', type=str, required=True)
        # parser.add_argument('--reduced_dimensions_count', type=int, required=True)
        # parser.add_argument('--classification_images_folder_path', type=str, required=True)
        # parser.add_argument('--classifier', type=str, required=True)

        return parser

logger = logging.getLogger(Task1.__name__)
logging.basicConfig(filename="logs/logs.log", filemode="w", level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S')

# if __name__=="main":
mypath="all/"

# Read all images

start_time = time.time()
image_reader = ImageReader()

train_images = image_reader.get_all_images_in_folder("all/")
train_images_names = image_reader.get_all_images_filenames_in_folder("all/")

train_all_labels = [label.split("-")[1] for label in train_images_names]
label_list = set()
label_list.update(train_all_labels)


test_images = image_reader.get_all_images_in_folder("100/100/")
test_images_names = image_reader.get_all_images_filenames_in_folder("100/100")

test_all_labels = [label.split("-")[1] for label in test_images_names]

print(len(test_images))

print(len(test_images[2].matrix))
# combined_images=[]
# combined_image_names=[]

combined_images = [*train_images,*test_images]

combined_image_names = [*train_images_names,*test_images_names]

print(len(combined_images))
print(combined_images[2].matrix)

feature_model="HOG"
dimensionality_reduction_technique="PCA"
k=10

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
image_feature_map = compute_image_feature_map(combined_image_names,rdfv)

print("calculated image feature map")

label_images_map = calculate_label_images_map(train_images_names,train_all_labels)
print("mapped label to images")

# time.time()
print("---- %s seconds " % (time.time() - start_time))

print("calculating random walk")
random_walk = compute_random_walk(image_feature_map,0.85)
print("random walk calculated..")

labelled_ppr=dict()


print("calculating seed and ppr for each label")
for lbl in label_list:
    teleportation_matrix = compute_seed_matrix(label_images_list=label_images_map[lbl],alpha=0.85)
    df = pd.DataFrame(ppr(teleportation_matrix,random_walk,0.85))

    df.insert(0,"Images",combined_image_names)
    labelled_ppr[lbl] = list(sorted(df.values,key=lambda x:x[1],reverse=True))


image_label_dict = associate_labels_to_test_images(test_images_names,labelled_ppr,label_list)
print("associated labels to test")

calculate_efficiency(image_label_dict,test_all_labels)
print("calculating efficiency")

# print(image_feature_map.keys())
# images_label_map = dict()



# Output().save_dict_as_json_file(image_feature_map, "test_tmp/combined_image_feature_map.json")




# transition_matrix = compute_similarity_matrix(image_feature_map)
# print(len(teleportation_matrix))
# print(teleportation_matrix)

# 4000x4000  4000x1             4000x1
#   T         P            S

# image_label_map = dict()