import sys
import argparse
import os
from utils.image_reader import ImageReader
from task_helper import TaskHelper
import numpy as np
from utils.constants import IMAGE_TYPE
from utils.feature_vector import FeatureVector
import pandas as pd  
from utils.constants import *
from utils.output import Output
from utils.dimensionality_reduction.kmeans import KMeans
from utils.dimensionality_reduction.lda import LatentDirichletAllocation
from utils.dimensionality_reduction.svd import SingularValueDecomposition
from utils.dimensionality_reduction.pca import PrincipalComponentAnalysis

class Task5:

    dst=[]
    partition_points = []
    bounds = []
    query_va = []
    images_va = []
    bounds = []
    dimension = 0
    distance_vector = None
    images_count = 0
    t = None
    def __init__(self):
        # parser = self.setup_args_parser()
        # self.args = parser.parse_args()
        pass
    def setup_args_parser(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--b', type=str, required=True)
        #parser.add_argument('--classifier', type=str, required=True) change to file path and take feature vector of the images
        return parser
    def feature_vector(self):
        image_reader = ImageReader()
        # training_images = image_reader.get_all_images_in_folder('D:\\MWDB\\4000') # 4800 images

        training_images = image_reader.get_all_images_in_folder('E:\\projects\\workspace\\1000\\1000')  # 4800 images

        # E:\projects\workspace

        # Step 2 - Extract feature vectors of all the training images n * m
        task_helper = TaskHelper()
        training_images = task_helper.compute_feature_vectors(
            'ELBP', 
            training_images)
        # Step 3 - Reduce the dimensions of the feature vectors of all the training images n * k
        training_images = task_helper.reduce_dimensions(
            'PCA', 
            training_images, 
            5)
        
        return training_images
    def VA_File(self,b,vectors):
        self.images_count = np.shape(vectors)[0]
        self.dimension = np.shape(vectors)[1]
        partition_points = [[]]*self.dimension
        bj = [0]*self.dimension
        for x in range(self.dimension):
            bj[x] = int(b/self.dimension)
            if x+1 <= b%self.dimension:
                bj[x]+=1
        for j in range(len(bj)):
            partition_points[j]=[0]*(pow(2,bj[j])+1)
        vectors = pd.DataFrame(vectors)
        new = vectors.copy()
        for x in range(np.shape(vectors)[1]):
            new[x],partition_points[x] = pd.cut(vectors[x],2**(bj[x]),labels=[bin(y)[2:].rjust(bj[x], '0') for y in range(2**(bj[x]))],retbins=True)
        self.partition_points = partition_points
        for x in range(len(new)):
            self.images_va.append(list(new.loc[x]))
        return new,partition_points

    def Generate_Output(self, size, va_file):
        output = {
            'Size of Index Structure': str(size) + ' bytes.',
            'Approximations': va_file,
        }
        return output

    def initialize_candidates_va_ssa(self):
        self.distance_vector = pd.DataFrame(data=[[0,sys.maxsize] for x in range(self.t)], columns=["index", "distance"])
        return sys.maxsize


    def candidate_va_ssa(self,d,i):
        n = self.t - 1
        # ans = [0]*len(self.distance_vector)
        if d<self.distance_vector.loc[n]['distance']:
            self.distance_vector.loc[n]['distance'] = d
            self.distance_vector.loc[n]['index'] = i
            # df = pd.DataFrame([ans,self.dst])
            # print(df)
            self.distance_vector = self.distance_vector.sort_values('distance',ascending='False')
        # print(self.distance_vector)
        return self.distance_vector.loc[n]['distance']

    def get_bounds(self,i):
        return self.bounds[i]

    def lp_metric(self,vi,vq,p):

        summation=0
        for i in range(len(vi)):
            summation+=pow(abs(vi[i] - vq[i]),p)
        return pow(summation,1/p)

    def va_ssa(self,vectors,vq,t):
        self.t = t
        d = self.initialize_candidates_va_ssa()
        search_results=[]
        for i in range(len(vectors)):
            l,_ = self.get_bounds(i)
            if l<d:
                print(l)
                d = self.candidate_va_ssa(self.lp_metric(vectors[i],vq,1),i)
                search_results.append(i)
        print(search_results)
        return search_results
    
    def getRecomputationMatrix(self,vectors):
        if 'right_factor_matrix' in vectors.keys():
            return vectors['right_factor_matrix']
        elif 'k_principal_components_eigen_vectors' in vectors.keys():
            return vectors['k_principal_components_eigen_vectors']
        elif 'components' in vectors.keys():
            return vectors['components']
        else: return vectors['centroids']
    
    def getReprojection(self,vectors,mat,comp):
        if 'right_factor_matrix' in vectors.keys():
            return SingularValueDecomposition().compute_reprojection(mat,comp)
        elif 'k_principal_components_eigen_vectors' in vectors.keys():
            return PrincipalComponentAnalysis().compute_reprojection(mat,comp)
        elif 'components' in vectors.keys():
            return LatentDirichletAllocation().compute_reprojection(mat,comp)
        else: return KMeans().compute_reprojection(mat,comp)
    
    def Generate_VA_Query_Image(self,values):
        for d in range(len(values)):
            if values[d]<self.partition_points[d][0]:
                self.query_va.append('000')
            elif values[d]>self.partition_points[d][7]:
                self.query_va.append('111')
            else:
                for val in range(len(self.partition_points[d])-1):
                    if values[d]>=self.partition_points[d][val] and values[d]<self.partition_points[d][val+1]:
                        self.query_va.append(format(val,'03b'))
        self.Generate_Bounds(values)
           

    def Generate_Bounds(self,vector):
        for x in self.images_va:
            lb = []
            ub = []
            for j in range(self.dimension):
                region = int(x[j],2)
                query_region = int(self.query_va[j],2)
                if region<query_region:
                    lb.append(vector[j] - self.partition_points[j][region+1])
                elif region==query_region:
                    lb.append(0)
                elif region>query_region:
                    lb.append(self.partition_points[j][region] - vector[j])
                if region<query_region:
                    ub.append(vector[j] - self.partition_points[j][region])
                elif region==query_region:
                    ub.append(max(vector[j] - self.partition_points[j][region],self.partition_points[j][region+1] - vector[j]))
                elif region>query_region:
                    ub.append(self.partition_points[j][region+1] - vector[j])
            self.bounds.append((sum(lb),sum(ub)))
        return self.bounds


def main():
    task = Task5()
    b=3
    images,vectors = task.feature_vector()
    # print(vectors)
    # comp = vectors['k_principal_components_eigen_vectors']
    comp = task.getRecomputationMatrix(vectors)
    # print(comp)
    reduced_feature_vector=vectors['reduced_dataset_feature_vector']
    k=np.shape(reduced_feature_vector)[1]
    bits_per_image=k*b
    va,partition_points=task.VA_File(bits_per_image,reduced_feature_vector)
    va_strings = [{images[x].filename:''.join(va.loc[x])} for x in range(len(va))]
    output = task.Generate_Output(len(images)*bits_per_image/8,va_strings)
    OUTPUT_FILE_NAME = 'output.json'
    # timestamp_folder_path = Output().create_timestamp_folder('D:\MWDB\CSE515-MWDB-Phase3\Submission\Outputs\Task5')  # /Outputs/Task1 -> /Outputs/Task1/2021-10-21-23-25-23

    timestamp_folder_path = Output().create_timestamp_folder('E:\\projects\\workspace\\CSE515-MWDB-Phase3\\Submission\\Outputs\\Task5')  # /Outputs/Task1 -> /Outputs/Task1/2021-10-21-23-25-23
    output_json_path = os.path.join(timestamp_folder_path, OUTPUT_FILE_NAME)
    Output().save_dict_as_json_file(output, output_json_path)
    bi = [bin(y)[2:].rjust(b, '0') for y in range(2**b)]
    #Get Query Image
    image = ImageReader().get_query_image('E:\\projects\\workspace\\test.png')
    # print(image)
    recomp = PrincipalComponentAnalysis().compute_reprojection(image.matrix.flatten(),comp)
    recomp = task.getReprojection(vectors,image.matrix.flatten(),comp)
    task.Generate_VA_Query_Image(recomp)
    result = task.va_ssa(reduced_feature_vector,recomp,10)
    result = [images[x].filename for x in result]
    print(result)
    # task.va_ssa(k, reduced_feature_vector, recomp,va)

if __name__ == "__main__":
    main()