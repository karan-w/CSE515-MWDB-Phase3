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
        training_images = image_reader.get_all_images_in_folder('D:\\MWDB\\4000') # 4800 images
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
        dimensions = np.shape(vectors)[1]
        partition_points = [[]]*dimensions
        bj = [0]*dimensions
        for x in range(dimensions):
            bj[x] = int(b/dimensions)
            if x+1 <= b%dimensions:
                bj[x]+=1
        for j in range(len(bj)):
            partition_points[j]=[0]*(pow(2,bj[j])+1)
        vectors = pd.DataFrame(vectors)
        new = vectors.copy()
        for x in range(np.shape(vectors)[1]):
            new[x],partition_points[x] = pd.cut(vectors[x],2**(bj[x]),labels=[bin(y)[2:].rjust(bj[x], '0') for y in range(2**(bj[x]))],retbins=True)
        return new,partition_points

    def Generate_Output(self, size, va_file):
        output = {
            'Size of Index Structure': str(size) + ' bytes.',
            'Approximations': va_file,
        }
        return output

    def initialize_candidates_va_ssa(self,k):
        self.dst=[]
        for i in range(k):
            self.dst[i]=sys.maxsize
        return sys.maxsize


    def candidate_va_ssa(self,d,i,n):

        ans = [0]*len(self.dst)
        if d<self.dst[n]:
            self.dst[n] = d
            df = pd.DataFrame(ans,self.dst)
            df = df.sort_values(self.dst,ascending=False)
        return self.dst[n]

    def get_bounds(self,ai,vq):
        # TODO implement upper and lower bound calc.
        return 1,2

    def lp_metric(self,vi,vq,p):

        summation=0
        for i in range(len(vi)):
            summation+=pow(abs(vi[i] - vq[i]),p)
        return pow(summation,1/p)
    
    def va_ssa(self,k,vectors,vq,a):
        d = self.initialize_candidates_va_ssa(k)
        search_results=[]
        for i in range(len(vectors)):
            l,_ = self.get_bounds(a[i],vq)
            if l<d:
                d = self.candidate_va_ssa(self.lp_metric(vectors[i],vq,1),i,k)
                search_results.append(d)

def main():
    task = Task5()
    b=3
    images,vectors = task.feature_vector()
    comp = vectors['k_principal_components_eigen_vectors']
    vectors=vectors['reduced_dataset_feature_vector']
    k=np.shape(vectors)[1]
    bits_per_image=k*b
    va,partition_points=task.VA_File(bits_per_image,vectors)

    va_strings = [{images[x].filename:''.join(va.loc[x])} for x in range(len(va))]
    output = task.Generate_Output(len(images)*bits_per_image/8,va_strings)
    OUTPUT_FILE_NAME = 'output.json'
    timestamp_folder_path = Output().create_timestamp_folder('D:\MWDB\CSE515-MWDB-Phase3\Submission\Outputs\Task5')  # /Outputs/Task1 -> /Outputs/Task1/2021-10-21-23-25-23
    output_json_path = os.path.join(timestamp_folder_path, OUTPUT_FILE_NAME)
    Output().save_dict_as_json_file(output, output_json_path)
    bi = [bin(y)[2:].rjust(b, '0') for y in range(2**b)]
    #Get Query Image
    image = ImageReader().get_query_image('D:\MWDB\\test.png')
    # print(image)
    recomp = PrincipalComponentAnalysis().compute_reprojection(image.matrix.flatten(),comp)
    # print(recomp)
    r=[]
    lower_bounds = []
    upper_bounds = []
    for i in range(k):
        for x in range(len(partition_points[i])-1):
            if (partition_points[i][x]<=recomp[i]) and (recomp[i]<partition_points[i][x+1]):
                r.append(bi[x])
    r = ''.join(r)
    print(r)

    task.va_ssa(k, vectors, recomp,va)

if __name__ == "__main__":
    main()