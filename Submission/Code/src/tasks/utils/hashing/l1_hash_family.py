from scipy.spatial.distance import cityblock

class L1HashFamily:
    def __init__(self, L, vectors):
        """
        Assuming that the vectors matrix will be m * k and k = L.
        Since we will create L hash functions, we need L row vectors which will
        be taken as the column vectors of the vectors matrix. 
        """
        self.hash_functions = []
        for vector in vectors.transpose(): # to get the column vector as row vector
            self.hash_functions.append(L1HashFunction(vector))

class L1HashFunction:
    def __init__(self, vector):
        self.vector = vector

    def hash(self, vector):
        return cityblock(self.vector, vector)

