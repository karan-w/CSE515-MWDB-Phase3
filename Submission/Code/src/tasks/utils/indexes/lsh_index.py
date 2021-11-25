from utils.hashing.l1_hash_family import L1HashFamily
import math
from scipy.spatial.distance import cityblock
import operator

class LSHIndex:
    def __init__(self, k, L, vectors, hash_family_type: str):
        self.k = k
        self.L = L
        if hash_family_type == "l1":
            hash_family = L1HashFamily(L, vectors)
            self.hash_functions = hash_family.hash_functions # List of L1HashFunction
        
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
            for image_filename in self.hash_tables[i][bucket_index]:
                retrieved_image_filenames_set.add(image_filename)

        images_hash_map = dict()
        for image in images:
            images_hash_map[image.filename] = image

        retrieved_images = []

        # run distacne function to find most siumilar t images
        for image_filename in retrieved_image_filenames_set:
            image = images_hash_map[image_filename]
            image.distance_from_query_image = cityblock(image.feature_vector, query_image_feature_vector)
            retrieved_images.append(image)

        sorted_retrieved_images = sorted(retrieved_images, key=operator.attrgetter('distance_from_query_image'))
        return sorted_retrieved_images[:t]
