from utils.hashing.l1_hash_family import L1HashFamily
import math
from scipy.spatial.distance import cityblock, cosine, euclidean
import operator
import sys

class LSHIndex:
    def __init__(self, k, L, vectors, hash_family_type: str, radius, distance_function_type: str):
        """
        distance_function_type: cosine, cityblock, euclidean
        """
        self.k = k
        self.L = L
        self.radius = radius
        self.distance_function_type = distance_function_type
        if hash_family_type == "l1":
            hash_family = L1HashFamily(10 * self.radius, vectors)
            self.hash_functions = hash_family.hash_functions # List of L1HashFunction
        
        self.bucketCount = 0
        self.overallImageCount = 0
        self.uniqueImageCount = 0
        self.hash_tables = []
        for i in range(self.L):
            self.hash_tables.append(dict())

    def populate_index(self, images):
        for image_id, image in enumerate(images):
            for i in range(self.L): # 0 .. L
                bucket_index = math.floor(self.hash_functions[i].hash(image.feature_vector)) % self.k 
                if bucket_index not in self.hash_tables[i]:
                    self.hash_tables[i][bucket_index] = [image.filename]
                else: 
                    self.hash_tables[i][bucket_index].append(image.filename)

    def get_similar_images(self, query_image_feature_vector, t, images):
        retrieved_image_filenames_set = set() # to prevent duplicate image filenames
        for i in range(self.L): # 0 .. L
            bucket_index = math.floor(self.hash_functions[i].hash(query_image_feature_vector)) % self.k 
            # if the bucket key exists in the hash table and that bucket is non empty
            if bucket_index in self.hash_tables[i] and len(self.hash_tables[i][bucket_index])>0:    
                self.bucketCount+=1
                self.overallImageCount+=len(self.hash_tables[i][bucket_index])
            for image_filename in self.hash_tables[i][bucket_index]:
                retrieved_image_filenames_set.add(image_filename)

        self.uniqueImageCount = len(retrieved_image_filenames_set)
        
        images_hash_map = dict()
        for image in images:
            images_hash_map[image.filename] = image

        retrieved_images = []

        # run distacne function to find most siumilar t images
        for image_filename in retrieved_image_filenames_set:
            image = images_hash_map[image_filename]
            if(self.distance_function_type == "cityblock"):
                image.distance_from_query_image = cityblock(image.feature_vector, query_image_feature_vector)
            elif(self.distance_function_type == "cosine"):
                image.distance_from_query_image = cosine(image.feature_vector, query_image_feature_vector)
            elif(self.distance_function_type == "euclidean"):
                image.distance_from_query_image = euclidean(image.feature_vector, query_image_feature_vector)
            retrieved_images.append(image)


        sorted_retrieved_images = sorted(retrieved_images, key=operator.attrgetter('distance_from_query_image'))
        return sorted_retrieved_images[:t]

    def get_size(self):
        total_size = 0
        for hash_table in self.hash_tables:
            total_size += sys.getsizeof(hash_table)

        for hash_function in self.hash_functions:
            total_size += sys.getsizeof(hash_function) 

        return total_size