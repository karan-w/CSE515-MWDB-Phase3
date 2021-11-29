
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
from sklearn.metrics import confusion_matrix

class PPR:

    args=None
    train_test_vectors=None
    start_time = None
    def __init__(self):
        pass

    def calculate_efficiency(self,pred, correct):
        count = 0

        # cf_matrix = confusion_matrix(list(pred.values()), list(correct.values()))
        ##print(cf_matrix)
        # false_positives = cf_matrix[0][1]
        # false_negatives = cf_matrix[1][0]
        # true_negatives = cf_matrix[1][1]
        # true_positives = cf_matrix[0][0]

        ##print("pred ",pred.values())
        ##print("correct ",correct.values())

        for key in pred:

            j, i = pred[key], correct[key]
            # ##print(i, "-", j)
            # if ".png" in i:
            #     i = i[:-4]
            # if ".png" in j:
            #     j = j[:-4]

            ##print(i, "-", j)
            if i == j:
                ##print("===========victory==============")
                count += 1
        ##print("correct ", count)
        rate = count / len(correct)
        ##print("eff. ", rate)

    def ppr(self,teleportation_matrix, random_walk, len_seed):

        # ##print("random walk ",random_walk)
        identity_matrix = np.identity(len(random_walk), dtype=float)
        inv = np.linalg.inv(identity_matrix - random_walk)
        pi = np.dot(inv, teleportation_matrix)

        pi = (pi - min(pi)) / (max(pi) - min(pi))

        # ##print("pi",pi)
        return pi
        # input()
        # P1_Teleportation_Discounting = np.zeros(len(teleportation_matrix))
        #
        # P1_ReSeeding_Value = (1 - alpha) / len_seed
        # for x in range(len(teleportation_matrix)):
        #     if teleportation_matrix[x][0]==0:
        #         P1_Teleportation_Discounting[x] = pi[x] / (alpha)
        #     else:
        #         P1_Teleportation_Discounting[x] = (pi[x] - P1_ReSeeding_Value) / (
        #                                               alpha)
        # ##print("p1 teleport discount ", P1_Teleportation_Discounting)
        #
        # P2_Value = P1_Teleportation_Discounting / sum(P1_Teleportation_Discounting)
        #
        # ##print("p2 val ", P2_Value)
        #
        # Seed_Set_Significance = 0
        # for x in range(len(teleportation_matrix)):
        #     Seed_Set_Significance += P2_Value[x]
        # P3_Value = P2_Value
        #
        #
        # for x in range(len(teleportation_matrix)):
        #     P3_Value[x] = P1_Teleportation_Discounting[x]
        #
        # ##print("p3 ", P3_Value)
        # return P3_Value
        # return pi

    def compute_image_feature_map(self,image_list, feature):
        img_feature_map = dict()
        for i, j in zip(image_list, feature):
            img_feature_map[i] = j.real.tolist()
        return img_feature_map

    def calculate_label_images_map(self,train_images_names, train_all_labels):

        label_images_map = dict()
        for i in range(len(train_all_labels)):
            if train_all_labels[i] in label_images_map:
                label_images_map[train_all_labels[i]].append(train_images_names[i])
            else:
                label_images_map[train_all_labels[i]] = [train_images_names[i]]
        return label_images_map

    def compute_seed_matrix(self,label_images_list, n, image_index_map, alpha=0.85):

        teleportation_matrix = [[0.0 for j in range(1)] for i in range(n)]

        # for i in range(len(label_images_list)):
        #     teleportation_matrix[i][0]=(1-alpha)/len(label_images_list)

        for i in label_images_list:
            teleportation_matrix[image_index_map[i]][0] = (1 - alpha) / len(label_images_list)

        teleportation_matrix_np = np.array(teleportation_matrix)
        return teleportation_matrix, teleportation_matrix_np
    def compute_random_walk(self,image_feature_map, alpha=0.85, kk=20):
        # random_walk = [[0.0 for i in range(len(image_feature_map))] for j in range(len(image_feature_map))]
        # random_walk = random_walk.to(device)
        random_walk = list()
        for image1, feature1 in image_feature_map.items():
            # ##print("img : ",image1)
            similar_list = []
            # ========================================================

            for image2, feature2 in image_feature_map.items():
                if image1 != image2:
                    # dist = distance.cityblock(feature1,feature2)
                    similar_list.append(distance.cityblock(feature1, feature2))
                else:
                    # dist=0
                    similar_list.append(0)
                    # ----------------------------------------------------------
                    # ==========================================================
            summ = sum(similar_list)
            similar_list = [(alpha * i) / summ for i in similar_list]
            random_walk.append(similar_list)
        return random_walk

    def associate_labels_to_test_images(self,test_images_names, labelled_ppr, label_list):

        ##print("-=========================================")
        # for ll in labelled_ppr.keys():
            #print("label ", ll)
            ##print(labelled_ppr[ll])
            ##print("---------------------------------")

        ##print("==========================================")
        # label_count=dict()
        query_index_in_label = dict()
        image_label_dict = dict()
        ##print("label list ", len(label_list))
        for name in test_images_names:
            ##print("associating for test image ", name)
            for l in label_list:
                # ##print("label ",l)
                query_index_in_label[l] = -1
                for k in range(len(labelled_ppr[l])):
                    # ##print("+++++++++++++++++++labelled ppr",labelled_ppr[l][k][0])
                    if labelled_ppr[l][k][0] == name:
                        query_index_in_label[l] = k
                        break
                ##print("index in label ppr of ", l, " for image ", name, " is ", query_index_in_label[l])
            mini = sys.maxsize
            labell = ""
            for lbl, indx in query_index_in_label.items():
                if mini > indx:
                    mini = indx
                    labell = lbl

            image_label_dict[name] = labell

        return image_label_dict

    def fit2(self,args):

        self.start_time = time.time()

        train_images = args["train_images"]

        train_images_names = [name.filename for name in train_images]

        ##print("train images name ............. ", len(train_images_names))

        train_all_labels = args["train_all_labels"]
        ##print("train label list............. ", len(train_all_labels))
        label_list = set()
        label_list.update(train_all_labels)
        ##print("label list............. ", len(label_list))


        test_images = args["test_images"]

        test_images_names = []
        for i in range(len(test_images)):
            splts = test_images[i].filename.split(".")
            test_images_names.append(splts[0] + "-test" + "." + splts[1])

        test_all_labels = args["test_all_labels"]

        test_image_name_label_map = dict()

        for i, j in zip(test_images_names, test_all_labels):
            test_image_name_label_map[i] = j

        ##print(len(test_images))

        ##print(len(test_images[2].matrix))
        # combined_images=[]
        # combined_image_names=[]

        test_labels_for_train = ["test"] * len(test_all_labels)

        combined_images = [*train_images, *test_images]

        combined_image_names = [*train_images_names, *test_images_names]

        # combined_image_names2 = [*train_images_names,*test_labels_for_train]
        combined_labels = [*train_all_labels, *test_all_labels]

        ##print("combined names ", combined_image_names)
        

        ##print("combined ",combined_labels)
        
        ##print(combined_images[2].matrix)

        # feature_model = args["feature_mo"]
        # dimensionality_reduction_technique = "PCA"
        # k = 20


        ##print("Features calculated")
        ##print("Dimensions reduced")

        rdfv = [*args["train_set_reduced_fv"],*args["test_set_reduced_fv"]]

        # ##print("hi")
        ##print(type(rdfv))
        ##print(len(rdfv))

        image_feature_map = self.compute_image_feature_map(combined_image_names, rdfv)
        ##print("calculated image feature map of length ", len(image_feature_map))
        ##print("combined length ", len(combined_image_names))

        label_images_map = self.calculate_label_images_map(train_images_names, train_all_labels)

        # label_images_map = calculate_label_images_map(combined_image_names,combined_labels)
        ##print("mapped label to images")

        # time.time()
        ##print("---- %s seconds " % (time.time() - self.start_time))

        ##print("calculating random walk")
        random_walk = self.compute_random_walk(image_feature_map, 0.85)
        ##print("random walk calculated.. for length ", len(random_walk))

        labelled_ppr = dict()

        image_index_map = dict()
        for i, img in enumerate(combined_image_names):
            image_index_map[img] = i

        ##print("calculating seed and ppr for each label")

        test_label_map = dict()
        for i in range(len(test_images)):
            teleportation_matrix, tele_np = self.compute_seed_matrix(label_images_list=[test_images_names[i]],
                                                                n=len(random_walk),
                                                                image_index_map=image_index_map, alpha=0.85)
            # df = pd.DataFrame(ppr(teleportation_matrix,random_walk,len(label_images_map[lbl]),0.15))


            pppr = self.ppr(tele_np, random_walk, 1)
            ##print("ppr ",pppr)

            # ##print("combined ", combined_labels)
            

            df = pd.DataFrame(pppr, columns=['ppr'])
            # df.insert(0, "Images", combined_image_names)
            df.insert(0, "Labels", combined_labels)
            # ##print("df \n",df)
            df = df.sort_values(by='ppr', ascending=False)

            # top_n_images = list(sorted(df.values, key=lambda x: x[1], reverse=True))[:49]

            if args["type"]=="X":
                top_n_images = list(df.iloc[:40]['Labels'])
            else:
                top_n_images = list(df.iloc[:20]['Labels'])
            # top_n_images = df.iloc[:49]['Labels'].

            ##print("top n ",top_n_images)
            
            test_count_map = dict()
            for lbl in top_n_images:
                if lbl in test_count_map.keys():
                    test_count_map[lbl] += 1
                else:
                    test_count_map[lbl] = 1

            # test_count = sorted(test_count_map.items(), key=lambda item: item[1], reverse=True)
            maxi = 0
            llbl = ""

            # ##print("test_map_count",test_count_map.values())
            # ##print("test_map_count keys ", test_count_map.keys())
            
            for k in test_count_map:
                if test_count_map[k] > maxi:
                    maxi = test_count_map[k]
                    llbl = k
                    # elif test_count_map[k]==maxi:
                    #     if k==test_images_names[i].split("-")[1]:
                    #         llbl = k
                    #         break

            ##print("llbl",llbl)
            test_label_map[test_images_names[i]] = llbl
            # test_label_map[test_images_names[i]] = test_count[0][0]


            # df = df.sort_values(by=["ppr"], ascending=False)
            # top_20_images = df[:20]
            #
            # labelled_ppr[lbl] = list(sorted(df.values, key=lambda x: x[1], reverse=True))

        # image_label_dict = associate_labels_to_test_images(test_images_names,labelled_ppr,label_list)
        ##print("associated labels to test")
        ##print("calculating efficiency")

        return list(test_label_map.values())
        # self.calculate_efficiency(test_label_map, test_image_name_label_map)

    def fit(self, args):

        start_time = time.time()
        image_reader = ImageReader()

        train_images = args["train_images"]
        train_images_names = [name.filename for name in train_images]

        print("train images name ............. ", len(train_images_names))

        train_all_labels = args["train_all_labels"]

        print("train label list............. ", len(train_all_labels))
        label_list = set()
        label_list.update(train_all_labels)
        print("label list............. ", len(label_list))

        # test
        test_images = args["test_images"]

        test_images_names=[]
        for i in range(len(test_images)):
            splts = test_images[i].filename.split(".")
            test_images_names.append(splts[0] + "-test" + "." + splts[1])

        test_all_labels = args["test_all_labels"]

        print(len(test_images))

        print(len(test_images[2].matrix))

        test_labels_for_train = ["test"] * len(test_all_labels)

        combined_images = [*train_images, *test_images]

        combined_image_names = [*train_images_names, *test_images_names]


        combined_labels = [*train_all_labels, *test_all_labels]

        print(len(combined_images))
        print(combined_images[2].matrix)

        feature_model = "HOG"
        dimensionality_reduction_technique = "PCA"
        k = 20

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
        image_feature_map = self.compute_image_feature_map(combined_image_names, rdfv)

        print("calculated image feature map of length ", len(image_feature_map))

        print("combined length ", len(combined_image_names))

        label_images_map = self.calculate_label_images_map(train_images_names, train_all_labels)

        # label_images_map = calculate_label_images_map(combined_image_names,combined_labels)
        print("mapped label to images")

        # time.time()
        print("---- %s seconds " % (time.time() - start_time))

        print("calculating random walk")
        random_walk = self.compute_random_walk(image_feature_map, 0.85)
        print("random walk calculated.. for length ", len(random_walk))

        labelled_ppr = dict()

        image_index_map = dict()
        for i, img in enumerate(combined_image_names):
            image_index_map[img] = i

        random_walk2 = np.array(random_walk)
        G = networkx.from_numpy_array(random_walk2)

        print("calculating seed and ppr for each label")
        for lbl in label_list:
            if lbl != "test":
                seeds = []
                for immg in label_images_map[lbl]:
                    seeds.append(image_index_map[immg])

                teleportation_matrix, tele_np = self.compute_seed_matrix(label_images_list=label_images_map[lbl],
                                                                    n=len(random_walk), image_index_map=image_index_map,
                                                                    alpha=0.85)
                # df = pd.DataFrame(ppr(teleportation_matrix,random_walk,len(label_images_map[lbl]),0.15))
                df = pd.DataFrame(self.ppr(tele_np, random_walk, len(label_images_map[lbl])))
                df.insert(0, "Images", combined_image_names)

                # subjects=[i+1 for i in range(len(teleportation_matrix))]
                #    #   ----xxxxx---- df3 = pd.DataFrame(Compute_Personalized_PageRank(subjects,random_walk, label_images_map[lbl]))
                # df3 = pd.DataFrame(Compute_Personalized_PageRank(subjects, random_walk, seeds))
                # df3.insert(0,"Images",combined_image_names)

                # personalization={}
                # for i,t in enumerate(teleportation_matrix):
                #     if t[0]!=0:
                #         personalization[i]=1
                #     else:
                #         personalization[i]=0
                #
                # pagerank_dict = networkx.pagerank(G,0.85,personalization=personalization)
                # pagerank_dict = dict(sorted(pagerank_dict.items()))
                #
                # df2 = pd.DataFrame(np.array(list(pagerank_dict.values())))
                # df2.insert(0,"Images",combined_image_names)

                # print("hi")
                labelled_ppr[lbl] = list(sorted(df.values, key=lambda x: x[1], reverse=True))

        image_label_dict = self.associate_labels_to_test_images(test_images_names, labelled_ppr, label_list)
        print("associated labels to test")

        print("calculating efficiency")


        # self.calculate_efficiency(image_label_dict, test_all_labels)
        return list(image_label_dict.values())

logger = logging.getLogger(PPR.__name__)
logging.basicConfig(filename="logs/logs.log", filemode="w", level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S')