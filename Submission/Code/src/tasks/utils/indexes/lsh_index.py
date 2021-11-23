from utils.hashing.l1_hash_family import L1HashFamily

class LSHIndex:
    def __init__(self, k, L, vectors, hash_family_type: str):
        self.k = k
        self.L = L
        self.vectors = vectors
        if hash_family_type == "l1":
            self.hash_family = L1HashFamily(L, vectors)

        self.hash_functions = self.hash_family.hash_functions # List of L1HashFunction
        self.hash_tables = [dict()] * L  # key = int (hashed value), value = list of image_filename (str)

    def populate_index(self, images_feature_vector, images_filename):
        for image_feature_vector in images_feature_vector:
            for i in range(self.L):
                bucket_index = self.hash_functions[i].hash(image_feature_vector) % self.k 
                if self.hash_tables[i].has_key(bucket_index):
                    self.hash_tables[i][bucket_index] = [images_filename[i]]
                else: 
                    self.hash_tables[i][bucket_index].append(images_filename[i])



