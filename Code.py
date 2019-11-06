#CODE


import pandas
import numpy as np
path     = "D:\\Keystroke\\keystroke.csv"
data     = pandas.read_csv(path)
subjects = data["subject"].unique()




Class ManhattanDetector:
    def evaluate(self):
        for subject in subjects:
            genuine_user_data = data.loc[data.subject == subject, 
                                    "H.period":"H.Return"]
            self.train        = genuine_user_data[:200]
            self.training()
 
    def training(self):
        self.mean_vector = self.train.mean().values
        
        


#genuine samples - remaining 200 from 400
self.test_genuine = genuine_user_data[200:]
 
#imposter samples
imposter_data = data.loc[data.subject != subject, :] 
self.test_imposter = imposter_data.groupby("subject").head(5).loc[:, 
                     "H.period":"H.Return"]       
                    



def testing(self):
    for i in range(self.test_genuine.shape[0]):
        cur_score = cityblock(self.test_genuine.iloc[i].values, 
                               self.mean_vector)
        self.user_scores.append(cur_score)
  
    for i in range(self.test_imposter.shape[0]):
        cur_score = cityblock(self.test_imposter.iloc[i].values, 
                               self.mean_vector)
        self.imposter_scores.append(cur_score)




eers = [] 
eers.append(evaluateEER(self.user_scores, self.imposter_scores))




#EER.py 
#Put in a separate file in same folder, since all detectors import it.
 
def evaluateEER(user_scores, imposter_scores):
    labels = [0]*len(user_scores) + [1]*len(imposter_scores)
    fpr, tpr, thresholds = roc_curve(labels, user_scores + imposter_scores)
    missrates = 1 - tpr
    farates = fpr
    dists = missrates - farates
    idx1 = np.argmin(dists[dists &gt;= 0])
    idx2 = np.argmax(dists[dists &lt; 0])
    x = [missrates[idx1], farates[idx1]]
    y = [missrates[idx2], farates[idx2]]
    a = ( x[0] - x[1] ) / ( y[1] - x[1] - y[0] + x[0] )
    eer = x[0] + a * ( y[0] - x[0] )
    return eer




#Keystroke_Manhattan.py
 
from scipy.spatial.distance import cityblock
import numpy as np
np.set_printoptions(suppress = True)
import pandas
from EER import evaluateEER
 
class ManhattanDetector:
  
    def __init__(self, subjects):
        self.user_scores = []
        self.imposter_scores = []
        self.mean_vector = []
        self.subjects = subjects
  
    def training(self):
        self.mean_vector = self.train.mean().values         
  
    def testing(self):
        for i in range(self.test_genuine.shape[0]):
            cur_score = cityblock(self.test_genuine.iloc[i].values, \
                                   self.mean_vector)
            self.user_scores.append(cur_score)
  
        for i in range(self.test_imposter.shape[0]):
            cur_score = cityblock(self.test_imposter.iloc[i].values, \
                                   self.mean_vector)
            self.imposter_scores.append(cur_score)
  
    def evaluate(self):
        eers = []
  
        for subject in subjects:        
            genuine_user_data = data.loc[data.subject == subject, \
                                         "H.period":"H.Return"]
            imposter_data = data.loc[data.subject != subject, :]
             
            self.train = genuine_user_data[:200]
            self.test_genuine = genuine_user_data[200:]
            self.test_imposter = imposter_data.groupby("subject"). \
                                 head(5).loc[:, "H.period":"H.Return"]
  
            self.training()
            self.testing()
            eers.append(evaluateEER(self.user_scores, \
                                     self.imposter_scores))
        return np.mean(eers)
 
path = "D:\\Keystroke\\keystroke.csv"
data = pandas.read_csv(path)
subjects = data["subject"].unique()
print "average EER for Manhattan detector:"
print(ManhattanDetector(subjects).evaluate())




#Keystroke_ManhattanFiltered.py
 
from scipy.spatial.distance import euclidean
class ManhattanFilteredDetector:
#just the training() function changes, rest all remains same.
    def training(self):
        self.mean_vector = self.train.mean().values
        self.std_vector = self.train.std().values
        dropping_indices = []
        for i in range(self.train.shape[0]):
            cur_score = euclidean(self.train.iloc[i].values, 
                                   self.mean_vector)
            if (cur_score &gt; 3*self.std_vector).all() == True:
                dropping_indices.append(i)
        self.train = self.train.drop(self.train.index[dropping_indices])
        self.mean_vector = self.train.mean().values
        
        
        

#Keystroke_ManhattanScaled.py
class ManhattanScaledDetector:
#training() and testing() change, rest all remains same.
    def training(self):
        #calculating mean absolute deviation deviation of each feature
        self.mean_vector = self.train.mean().values
        self.mad_vector  = self.train.mad().values
 
    def testing(self):
        for i in range(self.test_genuine.shape[0]):
            cur_score = 0
            for j in range(len(self.mean_vector)):
                cur_score = cur_score + \
                            abs(self.test_genuine.iloc[i].values[j] - \
                            self.mean_vector[j]) / self.mad_vector[j]
            self.user_scores.append(cur_score)
  
        for i in range(self.test_imposter.shape[0]):
            cur_score = 0
            for j in range(len(self.mean_vector)):
                cur_score = cur_score + \
                            abs(self.test_imposter.iloc[i].values[j] - \
                            self.mean_vector[j]) / self.mad_vector[j]
            self.imposter_scores.append(cur_score)
            
            
            
            
#Keystroke_SVM.py
 
from sklearn.SVM import OneClassSVM
 
class SVMDetector
#training() and testing() change, rest all remains same.
    def training(self):
        self.clf = OneClassSVM(kernel='rbf',gamma=26)
        self.clf.fit(self.train)
  
    def testing(self):
        self.u_scores = -self.clf.decision_function(self.test_genuine)
        self.i_scores = -self.clf.decision_function(self.test_imposter)
        self.u_scores = list(self.u_scores)
        self.i_scores = list(self.i_scores)
        
        
        
        
#keystroke_GMM.py
 
from sklearn.mixture import GMM
import warnings
warnings.filterwarnings("ignore")
 
class GMMDetector:
 
    def training(self):
        self.gmm = GMM(n_components = 2, covariance_type = 'diag', 
                        verbose = False )
        self.gmm.fit(self.train)
  
    def testing(self):
        for i in range(self.test_genuine.shape[0]):
            j = self.test_genuine.iloc[i].values
            cur_score = self.gmm.score(j)
            self.user_scores.append(cur_score)
  
        for i in range(self.test_imposter.shape[0]):
            j = self.test_imposter.iloc[i].values
            cur_score = self.gmm.score(j)
            self.imposter_scores.append(cur_score)
            
            
            
            
            
