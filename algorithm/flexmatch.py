# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 14:19:22 2021

@author: Vu Nguyen
"""

import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
#from scipy.stats import entropy
import random
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
#import ot  # ot
from scipy import stats
import time					
from .pseudo_labeling import Pseudo_Labeling




# FlexMatch Strategy for Pseudo-Labeling =======================================================================
# Zhang, Bowen, Yidong Wang, Wenxin Hou, Hao Wu, Jindong Wang, Manabu Okumura, and Takahiro Shinozaki. 
# "Flexmatch: Boosting semi-supervised learning with curriculum pseudo labeling." NeurIPS 2021
class FlexMatch(Pseudo_Labeling):
    # adaptive thresholding
    
    def __init__(self, unlabelled_data, x_test,y_test,num_iters=5,upper_threshold = 0.9, verbose = False):
        super().__init__( unlabelled_data, x_test,y_test,num_iters=num_iters,upper_threshold=upper_threshold,verbose=verbose)
        self.algorithm_name="FlexMatch"
    def predict(self, X):
        super().predict(X)
    def predict_proba(self, X):
        super().predict_proba(X)
    def evaluate(self):
        super().evaluate()
    def get_max_pseudo_point(self,class_freq,current_iter):
        return super().get_max_pseudo_point(class_freq,current_iter)
    def fit(self, X, y):
        print("=====",self.algorithm_name)
        self.nClass=len(np.unique(y))
        if len(np.unique(y)) < len(np.unique(self.y_test)):
            print("num class in training data is less than test data !!!")
            
        self.num_augmented_per_class=[0]*self.nClass
        
        self.label_frequency=self.estimate_label_frequency(y)

        for current_iter in (tqdm(range(self.num_iters)) if self.verbose else range(self.num_iters)):

            # Fit to data
            self.model.fit(X, y.ravel())
            
            self.evaluate()

            # estimate prob using unlabelled data
            pseudo_labels_prob = self.model.predict_proba(self.unlabelled_data)
            
            num_points=pseudo_labels_prob.shape[0]        
            
            #go over each row (data point), only keep the argmax prob
            max_prob=[0]*num_points
            
            max_prob_matrix=np.zeros((pseudo_labels_prob.shape))
            for ii in range(num_points): 
                idxMax=np.argmax(pseudo_labels_prob[ii,:])
                
                max_prob_matrix[ii,idxMax]=pseudo_labels_prob[ii,idxMax]
                max_prob[ii]=pseudo_labels_prob[ii,idxMax]
        
        
            # for each class, count the number of points > threshold
            countVector=[0]*self.nClass
            for cc in range(self.nClass):
                idx_above_threshold=np.where(max_prob_matrix[:,cc]>self.upper_threshold)[0]
                countVector[cc]= len( idx_above_threshold ) # count number of unlabeled data above the threshold
            countVector_normalized=np.asarray(countVector)/np.max(countVector)
            
            if self.verbose:
                print("class threshold:", np.round(countVector_normalized*self.upper_threshold,2))
            
            augmented_idx=[]
            for cc in range(self.nClass):
                # compute the adaptive threshold for each class
                class_upper_thresh=countVector_normalized[cc]*self.upper_threshold

                MaxPseudoPoint=self.get_max_pseudo_point(self.label_frequency[cc],current_iter)
                idx_sorted = np.argsort( max_prob_matrix[:,cc])[::-1][:MaxPseudoPoint] # decreasing        

                idx_above_threshold = np.where(max_prob_matrix[idx_sorted,cc] > class_upper_thresh)[0]
                labels_within_threshold= idx_sorted[idx_above_threshold]
                augmented_idx += labels_within_threshold.tolist()

                X,y = self.post_processing(cc,labels_within_threshold,X,y)
                
                
                
            if self.verbose:
                print("#augmented:", self.num_augmented_per_class, " len of training data ", len(y))
          

            if np.sum(self.num_augmented_per_class)==0: # no data point is augmented
                return self.test_acc
                
            # remove the selected data from unlabelled data
            self.unlabelled_data = np.delete(self.unlabelled_data, np.unique(augmented_idx), 0)
                
        # evaluate at the last iteration for reporting purpose
        self.model.fit(X, y.ravel())

        self.evaluate()
        return self.test_acc
    

    