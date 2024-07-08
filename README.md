# material_science

%%capture
import torch
!pip install pandas
import pandas as pd
!pip install numpy
import numpy as np
!pip install scipy
from scipy.stats import bernoulli
!pip install matplotlib
import matplotlib.pyplot as plt
!pip install scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
!pip install qpsolvers
import qpsolvers
from qpsolvers import solve_qp
!pip install qpsolvers[cvxopt]
!pip install qpsolvers[open_source_solvers]
!pip install qpsolvers[clarabel]


raisin_whole_data = pd.read_csv('raisin.csv')
raisin_whole_data.columns = ['area','majlength','minlength','eccentric','cvx',
                             'extent','perimeter','class']
class_to_keep = raisin_whole_data['class'].copy()

raisin_whole_data = raisin_whole_data.drop(['class'], axis=1)
raisin_whole_data = StandardScaler().fit_transform(X=raisin_whole_data)

raisin_whole_data = pd.DataFrame(raisin_whole_data)
raisin_whole_data['class'] = class_to_keep
raisin_whole_data.columns = ['area','majlength','minlength','eccentric','cvx',
                             'extent','perimeter','class']
raisin_whole_data = raisin_whole_data.replace('Besni',-1)
raisin_whole_data = raisin_whole_data.replace('Kecimen',1)
raisin = raisin_whole_data.sample(frac=0.8)
list_index_test = []
for i in raisin_whole_data.index:
    if i not in raisin.index:
        list_index_test.append(i)
raisin_test = raisin_whole_data.filter(items=list_index_test, axis=0)



n_samples = raisin.shape[0]
n_samples_test = raisin_test.shape[0]

raisin = raisin.reset_index(drop=True)
raisin_test = raisin_test.reset_index(drop=True)

def rbf(x,y,l=1):
    """Gaussian kernel

    Parameters
    -------------------------------
    x : float
    a real number

    y : float
    a real number

    l: float, non zero
    a scale parameter
    -------------------------------
    """
    dim = x.shape[0]
    vect = np.empty(dim)
    if dim == y.shape[0]  :
        d = np.exp((-1)*((np.linalg.norm(x-y))/(2*(l**2))))
        return d
    else :
        for i in range(dim):
            vect[i] = np.exp((-1)*(np.linalg.norm(x[i] - y))/(2*(l**2)))
        return vect





label = np.zeros(n_samples)
for i in range(n_samples):
    random = bernoulli.rvs(p=3/4)
    if raisin.loc[i,'class'] == 1 and random == 0:
        label[i] = 1
    else:
        label[i] = -1
raisin['label'] = label



svm_train = SVC(kernel='sigmoid', probability = True).fit(X=raisin.to_numpy()[:,:-2], y=raisin.to_numpy()[:,-1])


probas = svm_train.predict_proba(raisin.to_numpy()[:,:-2])


proba_gap = np.zeros(n_samples)
for i in range(n_samples):
    proba_gap[i] = probas[i,1] - probas[i,0]


raisin['proba_gap'] = proba_gap


l_boundary = raisin[raisin['label'] == 1]['proba_gap'].min()


relab = np.empty(n_samples)
for i in range(n_samples):
    if raisin.loc[i,'proba_gap'] < l_boundary:
        relab[i] = -1
    elif raisin.loc[i,'label'] == 1 or raisin.loc[i,'proba_gap'] >= 0:
        relab[i] = 1
    else:
        relab[i] = 0
raisin['relab'] = relab


B=1000
labeled_data = raisin[raisin['relab'] != 0].copy()
output_labeled = labeled_data['relab'].to_numpy()
list_of_index = labeled_data.index
labeled_data = labeled_data.reset_index(drop=True)
labeled_data = labeled_data.to_numpy()[:,:-4]
unlabeled_data = raisin.drop(index=list_of_index,axis=0)
unlabeled_data = unlabeled_data.to_numpy()[:,:-4]
n_unlabeled = unlabeled_data.shape[0]
n_labels = labeled_data.shape[0]
capital_k = np.zeros((n_labels,n_labels))
kappa = np.zeros(n_labels)



#construction of capital_k
for i in range(n_labels):
    for j in range(i,n_labels):
        capital_k[i,j] = rbf(x=labeled_data[i,:],y=labeled_data[j,:])

capital_k = capital_k + capital_k.T
for i in range(n_labels):
    capital_k[i,i] = 1

capital_k[np.where(np.isnan(capital_k) == True)] = 0

#construction of kappa
ratio_lab_unlab = n_labels/n_unlabeled

for i in range(n_labels):
    vector = np.empty(n_unlabeled)
    for k in range(n_unlabeled):
        vector[k] = rbf(x=labeled_data[i,:],y=unlabeled_data[k,:])    
    kappa[i] = ratio_lab_unlab*np.sum(vector)

kappa = -kappa



ones_transposed = np.ones(n_labels).reshape(1,n_labels)
a_mat = np.vstack((ones_transposed,ones_transposed*-1,
                   np.eye(n_labels),np.eye(n_labels)*-1))
epsilon = (np.sqrt(n_labels)-1)/np.sqrt(n_labels)
ub_mat = np.vstack((n_labels*(1+epsilon),n_labels*(epsilon-1),
                    np.ones(n_labels).reshape(n_labels,1)*B,
                    np.zeros(n_labels).reshape(n_labels,1)))



beta_opti = solve_qp(P=capital_k,q=kappa,G=a_mat,h=ub_mat,solver='cvxopt')


svm_weighted = SVC().fit(X=labeled_data,y=output_labeled,sample_weight=beta_opti)

predictions_weighted = svm_weighted.predict(raisin_test.to_numpy()[:,:-1])

positive = 0
true_positive = 0
for i in range(n_samples_test):
    if predictions_weighted[i] == 1:
        positive += 1
        if raisin.loc[i,'class'] == 1:
            true_positive += 1
print(true_positive/positive)

