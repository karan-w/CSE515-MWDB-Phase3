import sys
import argparse
import os
import json
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
from utils.feature_models.hog import HistogramOfGradients
from utils.feature_models.elbp import ExtendedLocalBinaryPattern
from utils.feature_models.cm import ColorMoments

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
    bits = 0
    data = []
    images = None
    recomp = None

    unique_buckets=set()
    unique_images=set()

    def __init__(self,args=None):
        if args is None:
            parser = self.setup_args_parser()
            self.args = parser.parse_args()
        else:
            self.args = args
    def setup_args_parser(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--b', type=int, required=True)
        parser.add_argument('--feature_model', type=str, required=True)
        parser.add_argument('--t', type=int, required=True)
        parser.add_argument('--k', type=int, required=True)
        parser.add_argument('--output_folder_path', type=str, required=True)
        parser.add_argument('--images_folder_path', type=str, required=True)
        parser.add_argument('--latent_semantics_file', type=str, required=True)
        parser.add_argument('--query_image_path', type=str, required=True)
        parser.add_argument('--dimensionality_reduction_technique', type=str, required=True)
        parser.add_argument('--generate_va_file', type=str, required=False)
        return parser
    def feature_vector(self):
        #Compute feature vectors for all the images in the given folder
        image_reader = ImageReader()
        training_images = image_reader.get_all_images_in_folder(self.args.images_folder_path) # 4800 images
        task_helper = TaskHelper()
        training_images = task_helper.compute_feature_vectors(
            self.args.feature_model, 
            training_images)
        
        return training_images
    def read_latent_semantics_file(self):
        # Read the latent semantics file 
        self.data = json.load(open(os.path.join(self.args.latent_semantics_file,'{0}-{1}-{2}.json'.format(self.args.feature_model,self.args.k,self.args.dimensionality_reduction_technique))))

    def compute_feature_vector(self, image):
        #Compute the feature vector for an image
        if self.args.feature_model == COLOR_MOMENTS:
            return ColorMoments().get_color_moments_fd(image)
        elif self.args.feature_model == EXTENDED_LBP:
            return ExtendedLocalBinaryPattern().get_elbp_fd(image)
        elif self.args.feature_model == HISTOGRAM_OF_GRADIENTS:
            return HistogramOfGradients().get_hog_fd(image)
        else:
            raise Exception(f"Unknown feature model - {self.args.feature_model}")

    def VA_File(self):
        # Compute the VA-File Index Structure for the given dataset
        vectors = self.images
        self.images_count = np.shape(self.images)[0]
        self.dimension = np.shape(self.images)[1]
        b=self.args.b*self.args.k #Total number of bits assigned to each image of the dataset
        partition_points = [[]]*self.dimension #partition points for each dimension
        bj = [0]*self.dimension #Number of bits to be used for each dimension
        # Calculate the size of the partition points for each dimension
        for x in range(self.dimension):
            bj[x] = int(b/self.dimension)
            if x+1 <= b%self.dimension:
                bj[x]+=1
        # Partition points are assigned the size of 2^(bits in that dimension + 1)
        for j in range(len(bj)):
            partition_points[j]=[0]*(pow(2,bj[j])+1)
        # The vectors are stored in a pandas dataframe
        vectors = pd.DataFrame(vectors)
        new = vectors.copy()
        # Each dimension is divided into regions that have the size of partition points in that dimension and each dimension of each image is assigned a binary value of the region it liews in  
        for x in range(self.dimension):
            new[x],partition_points[x] = pd.cut(vectors[x],2**(bj[x]),labels=[bin(y)[2:].rjust(bj[x], '0') for y in range(2**(bj[x]))],retbins=True)
        self.partition_points = partition_points
        # Storing VA-File strings of each image
        for x in range(len(new)):
            self.images_va.append(list(new.loc[x]))
        return new,partition_points

    def Generate_Output(self, size, va_file,result):
        output = {
            'Size of Index Structure': str(size) + ' bytes.',
            'Approximations': va_file,
            '{0} Most Similar Images'.format(self.args.t):['{0} -> Distance : {1}'.format(result[x].filename,result[x].distance_from_query_image) for x in range(len(result))],
            'Unique Buckets Searched': len(self.unique_buckets),
            'Buckets':self.unique_buckets,
            'Unique Images Searched': len(self.unique_images)
        }
        return output

    def initialize_candidates_va_ssa(self):
        # initialize the vector that will have the t most similar images to the query image with index 0 and distance having maxInt 
        self.distance_vector = pd.DataFrame(data=[[0,sys.maxsize] for x in range(self.args.t)], columns=["index", "distance"])
        return sys.maxsize


    def candidate_va_ssa(self,d,i):
        # If a candidate is found, replace the least similar image from the vector of size t with the new candidate 
        n = self.args.t - 1
        if d<self.distance_vector.iloc[n]['distance']:
            self.distance_vector.iloc[n]['distance'] = d
            self.distance_vector.iloc[n]['index'] = i
            self.distance_vector = self.distance_vector.sort_values('distance')
        return self.distance_vector.iloc[n]['distance']

    def get_bounds(self,i):
        # returns the lower and upeer bound of the ith image
        return self.bounds[i]

    def lp_metric(self,vi,vq,p):
        # Computes the distance between 2 vectors
        summation=0
        for i in range(len(vi)):
            summation+=pow(abs(vi[i] - vq[i]),p)
        return pow(summation,1/p)

    def va_ssa(self,vectors,vq):
        # Initialize the array of t with the base values and assign the maximum integer value to d
        d = self.initialize_candidates_va_ssa() 
        # for all images in th folder check if the distance between image vector and the query vector become a candidate, if yes assign this newly calculated distance to d 
        for i in range(len(vectors)):
            l,_ = self.get_bounds(i) # l is the ower bound of image i
            if l<d: # if lower bound is less than the current distance, this image is a candidate
                d = self.candidate_va_ssa(self.lp_metric(vectors[i],vq,1),i)
                self.unique_buckets.add(self.va_strings[vectors[i]])
                self.unique_images.add(vectors[i])

        return self.distance_vector
    
    def getRecomputationMatrix(self,vectors):
        # Return the appropriate vector for query image reprojection on the feature space
        if 'right_factor_matrix' in vectors.keys():
            return vectors['right_factor_matrix']
        elif 'k_principal_components_eigen_vectors' in vectors.keys():
            return vectors['k_principal_components_eigen_vectors']
        elif 'components' in vectors.keys():
            return vectors['components']
        else: return vectors['centroids']
    
    def getReprojection(self,vectors,mat,comp):
        # Reproject the query image on the feature space
        if 'right_factor_matrix' in vectors.keys():
            return SingularValueDecomposition().compute_reprojection(mat,comp)
        elif 'k_principal_components_eigen_vectors' in vectors.keys():
            return PrincipalComponentAnalysis().compute_reprojection(mat,comp)
        elif 'components' in vectors.keys():
            return LatentDirichletAllocation().compute_reprojection(mat,comp)
        else: return KMeans().compute_reprojection(mat,comp)
    
    def Generate_VA_Query_Image(self,values):
        # Assign bits for all the dimensions according to the partition points of the dimension
        for d in range(len(values)):
            # Dimension : d
            if values[d]<self.partition_points[d][0]:
                self.query_va.append('0'*self.bits)
            elif values[d]>self.partition_points[d][7]:
                self.query_va.append('1'*self.bits)
            else:
                # Loop over all the partition points of dimension d and assign region for that dimension of the query image
                for val in range(len(self.partition_points[d])-1):
                    if values[d]>=self.partition_points[d][val] and values[d]<self.partition_points[d][val+1]:
                        self.query_va.append(format(val,'0{0}b'.format(self.bits)))
        # Generate lower and upper bounds of each image in the database with the query image
        self.Generate_Bounds(values)
           

    def Generate_Bounds(self,vector):
        # Calculate lower and upper bounds each image 
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

    def reproject_query_image(self):
        # Read query image
        image = ImageReader().get_query_image(self.args.query_image_path)
        # Project query image on the feature space
        self.recomp = self.getReprojection(self.data['drt_attributes'],self.compute_feature_vector(image.matrix),self.comp)
        # Generate VA string for the query image
        self.Generate_VA_Query_Image(self.recomp)

    def project_images(self):
        # Get the dimensionality reduction technique m*k matrix
        self.comp = self.getRecomputationMatrix(self.data['drt_attributes'])
        new_vector = []
        for x in self.images:
            # Prooject each image in the database on the latent semantic file
            new_vector.append(self.getReprojection(self.data['drt_attributes'],x.feature_vector,self.comp))
        return new_vector

    def generate_ouput(self,final):
        # Generate output
        output = self.Generate_Output(len(self.images)*(self.args.b*self.args.k)/8,self.va_strings,final)
        OUTPUT_FILE_NAME = 'output.json'
        timestamp_folder_path = Output().create_timestamp_folder(self.args.output_folder_path)  # /Outputs/Task1 -> /Outputs/Task1/2021-10-21-23-25-23
        output_json_path = os.path.join(timestamp_folder_path, OUTPUT_FILE_NAME)
        Output().save_dict_as_json_file(output, output_json_path)

    def get_similar_images(self,images=None):
        # Read latent semantic file 
        self.read_latent_semantics_file() 
        if images==None:
            self.images = self.feature_vector()
        else: self.images = images
        self.original_images = self.images.copy()
        # Project all the images on the latent space
        self.images = self.project_images()
        #Compute the VA-Files
        va,_ = self.VA_File()
        # Compute VA strings for all images : format --> filename : k*b bits representation
        self.va_strings = {self.original_images[x].filename:''.join(va.loc[x]) for x in range(len(va))}
        # Project query image on the latent space
        if self.recomp==None:
            self.reproject_query_image()
        # Compute the t most similar images with respect to query image in the database
        result = self.va_ssa(self.images,self.recomp)
        final = [self.original_images[result.iloc[x]['index']] for x in range(len(result))]
        # for t most similar images, store its distance from the query image
        for x in range(len(final)):
            final[x].distance_from_query_image = result.iloc[x]['distance']
        return final
    def execute(self):
        result = self.get_similar_images()
        self.generate_ouput(result)

def main():
    task = Task5()
    task.execute()


if __name__ == "__main__":
    main()