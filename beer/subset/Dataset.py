'''
Created on April 8, 2016

@author: jianmo
'''
import scipy.sparse as sp
import numpy as np
import cPickle as pickle

class Dataset(object):
    '''
    classdocs
    '''

    def __init__(self, path):
        '''
        Constructor
        '''
        self.trainMatrix = self.load_rating_file_as_matrix(path + ".train.rating")
        self.testRatings = self.load_rating_file_as_list(path + ".test.rating")
        self.testNegatives = self.load_negative_file(path + ".test.negative")
        assert len(self.testRatings) == len(self.testNegatives)

        self.validRatings = self.load_rating_file_as_list(path + ".valid.rating")
        self.validNegatives = self.load_negative_file(path + ".valid.negative")
        assert len(self.validRatings) == len(self.validNegatives)
        
        self.testAllNegatives = self.load_negative_file(path + ".test.all.negative")
        self.validAllNegatives = self.load_negative_file(path + ".valid.all.negative")

        self.num_users, self.num_items = self.trainMatrix.shape
        
    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList
    
    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1: ]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList
    
    def load_rating_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        with open('data_dir/beer/encodings.pkl', 'rb') as fp:
            nUser,nItem,user_indices,indices_user,item_indices,indices_item,chars,char_indices,indices_char = pickle.load(fp)
        num_users = nUser
        num_items = nItem
        '''
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        '''
        # Construct matrix
        #mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        mat = sp.dok_matrix((num_users, num_items), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if (rating > 0):
                    mat[user, item] = 1.0
                line = f.readline()    
        return mat
