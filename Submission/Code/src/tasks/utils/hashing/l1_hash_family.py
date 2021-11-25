class L1HashFamily:
    def __init__(self, w, vectors):
        """
        Assuming that the vectors matrix will be m * k and k = L.
        Since we will create L hash functions, we need L row vectors which will
        be taken as the column vectors of the vectors matrix. 
        """
        self.w = w
        self.hash_functions = []

        for vector in vectors.transpose(): # to get the column vector as row vector
            self.hash_functions.append(L1HashFunction(self.w, vector))

class L1HashFunction:
    def __init__(self, w, vector):
        self.w = w
        self.vector = vector

    def hash(self, vector):
        return sum([int(vector[i]-s/self.w) for i, s in enumerate(self.vector)])

