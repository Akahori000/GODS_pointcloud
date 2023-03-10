#Copyright (c) 2023 Mitsubishi Electric Research Laboratories (MERL). All rights reserved.
#
#The software, documentation and/or data in this file is provided on an "as is" basis, 
#and MERL has no obligations to provide maintenance, support, updates, enhancements or
# modifications. MERL specifically disclaims any warranties, including, but not limited to, 
# the implied warranties of merchantability and fitness for any particular purpose. In no event 
# shall MERL be liable to any party for direct, indirect, special, incidental, or consequential damages, 
# including lost profits, arising out of the use of this software and its documentation, even 
# if MERL has been advised of the possibility of such damages.
#
#As more fully described in the license agreement that was required in order to download 
#this software, documentation and/or data, permission to use, copy and modify this 
#software without fee is granted, but only for educational, research and non-commercial purposes.

""" Kernelized One-Class Discriminative Subspaces (KODS) for anomaly detection. 
    Implementation in Python using PyManOpt. 
    implemented by Anoop Cherian and Jue Wang.
"""
import autograd.numpy as np
from pymanopt import Problem
from pymanopt.solvers import ConjugateGradient, TrustRegions
from pymanopt.manifolds import Product, Stiefel, Euclidean, Sphere, Oblique, GeneralizedStiefel
import argparse
import os
import random
import pickle
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix, auc
from sklearn.metrics.pairwise import chi2_kernel
from scipy.spatial.distance import cdist
import get_features as getf
import pandas as pd


def compute_chisq_kernel(A, B, sigma):
    """
        chi-squared kernel
    """
    kernel = chi2_kernel(A, B, gamma=sigma)
    return kernel

def compute_min_kernel(A, B, k):
    """
        histogram intersection kernel: min kernel: generalized histogram intersection kernel.
    """
    kernel = np.zeros((A.shape[0], B.shape[0]))
    for k in range(A.shape[1]):
        kernel += np.minimum(A[:,k][:,np.newaxis], B[:,k][:,np.newaxis].transpose())
    kernel = kernel/A.shape[1]
    return kernel
        
def compute_kernel(A, B, sigma, num_pts, k, same=False):
    # if args.kernel == 'linear':
    #     kernel = np.matmul(A, B.transpose()) # linear kernel
    # elif args.kernel == 'rbf':
    #     kernel = np.exp(-(cdist(A, B, 'sqeuclidean'))/(2.0*sigma)) # RBF kernel
    # elif args.kernel == 'chisq':
    #     kernel = compute_chisq_kernel(A, B, sigma) # chi-sq kernel
    # elif args.kernel == 'min':
    #     kernel = compute_min_kernel(A, B, k) # histogram intersection kernel.
    kernel = np.exp(-(cdist(A, B, 'sqeuclidean'))/(2.0*sigma)) # RBF kernel

    if same == True:
        kernel = (kernel + kernel.transpose())/2.0 # make sure the kernel is symmetric, else complex roots may come up.
        kernel += np.eye(num_pts)*1e-7 # regulrize the kernel, avoid low-rank for numerical problems.
    #print('mean of the kernel=%f' % (kernel.mean()))
    return kernel

# compute prediction accuracy.
def calculate_accruacy(opt, data, label_gt, eta, K):
    label = []
    Y, Z = opt # this is n x kv 
    Y = Y*Y  # this is nxk
    Z = Z*Z

    # estimate b1 and b2.
    b1 = (eta - np.matmul(K, Z)).max(0) 
    b2 = (eta + np.matmul(K, Y)).min(0)
    
    # kernel for test set.
    KK_test = compute_kernel(data, X, sigma, num_pts, k)
    
    # classify each test point as positive or negative.
    for i in range(len(data)):
        ww1 = np.matmul(KK_test[i,:][np.newaxis, :], Z) + b1 
        ww2 = -np.matmul(KK_test[i,:][np.newaxis, :], Y) + b2
        
        # we use exact thresholds for judging accuracy. That is, if the pose falls between the two subspaces, 
        # we take that as normal class, else abnormal.
        if np.all(ww1 > 0) and np.all(ww2 < 0):
            label.append(1)
        else:
            label.append(0)
            
    print("predicted_label", label)       
    accuracy = accuracy_score(label_gt, label)
    F1 = f1_score(label_gt, label)        
    precision, recall, fbeta_score, support = precision_recall_fscore_support(label_gt, label)
    return accuracy, F1, precision, recall

# compute prediction accuracy.
def calculate_AUC(opt, data, label_gt, sigma, eta, K, k):
    ftr_roc_auc = 0
    tpr = []
    fpr = []
    cls0 = []
    cls1 = []
    thresh_arr = []

    for j in range(-200, 200, 1):
        thresh = j*0.002
        thresh_arr.append(round(thresh,4))
        label = []
        Y, Z = opt # this is n x kv 
        Y = Y*Y  # this is nxk
        Z = Z*Z

        # estimate b1 and b2.
        b1 = (eta - np.matmul(K, Z)).max(0) 
        b2 = (eta + np.matmul(K, Y)).min(0)
        
        # kernel for test set.
        KK_test = compute_kernel(data, X, sigma, num_pts, k, same=False)
        
        # classify each test point as positive or negative.
        for i in range(len(data)):
            ww1 = np.matmul(KK_test[i,:][np.newaxis, :], Z) + b1 
            ww2 = -np.matmul(KK_test[i,:][np.newaxis, :], Y) + b2
            
            # we use exact thresholds for judging accuracy. That is, if the pose falls between the two subspaces, 
            # we take that as normal class, else abnormal.
            if np.all(ww1 > thresh) and np.all(ww2 < -(thresh)):
                label.append(1)
            else:
                label.append(0)
                
        # 0???1?????????
        label_t = [1 if n == 0 else 0 for n in label_gt]
        result = [1 if n == 0 else 0 for n in label]

        #print("1:", sum(result), "0:", (len(result) - np.sum(result)))
        cls1.append(sum(result))
        cls0.append((len(result) - np.sum(result)))

        cm = confusion_matrix(label_t, result)
        #print(cm)
        tn, fp, fn, tp = cm.flatten()
        tpr.append(tp)
        fpr.append(fp)

        # ?????????????????????   
        accuracy = accuracy_score(label_gt, label)
        F1 = f1_score(label_gt, label)        
        precision, recall, fbeta_score, support = precision_recall_fscore_support(label_gt, label)
        
    if (min(tpr) == 0 and min(fpr) == 0 and max(cls1) == len(label_gt) and max(cls0) == len(label_gt)):
        fpr = fpr / np.max(fpr)
        tpr = tpr / np.max(tpr)
        ftr_roc_auc = auc(fpr, tpr)
        print(ftr_roc_auc)
        print("thresh", thresh_arr)
        print("num of cls0", cls0)
        print("num of cls1", cls1)

        plt.figure(figsize=(5,5))
        plt.plot(fpr, tpr)
        plt.xlabel('FPR: False Positive Rete', fontsize = 13)
        plt.ylabel('TPR: True Positive Rete', fontsize = 13)
        plt.grid()
        plt.savefig(dir + str(k) + "/auc_" + anomaly_object + "_subdim_" + str(k) + "_auc_" + str(round(ftr_roc_auc,3)) + ".png")
    return ftr_roc_auc #accuracy, F1, precision, recall

seed = 123
print('seed=%d'%(seed))
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='GODS: pose anomaly detection.')
    parser.add_argument('--split_num', default=0, type=int, help='in case of cross-validation, specfiy the cross-validation number (1/2/3/4)')    
    parser.add_argument('--num_subspaces', default=2, type=int, help='number of hyperplanes in each of GODS subspaces. Default=2')    
    parser.add_argument('--eta', default=0.01, type=float, help='eta, controlling how far are the predictions from the hyperplanes. Default=0.01')        
    parser.add_argument('--sigma', default=1., type=float, help='sigma, std for the rbf kernel, Default=1.0') 
    parser.add_argument('--max_iter', default=100, type=int, help='max-number of Riemannian optimization iterations. Default=100')    
    parser.add_argument('--optim', default='cg', type=str, help='Optimizer to use. conjugate-gradient (cg) / trust-regions (tr). Default=cg')    
    parser.add_argument('--thresh', default=0., type=float, help='threshold for deciding if the given data point is within the two subspaces. Default=0')    
    parser.add_argument('--unnormalize', action='store_true', help='Should you normalize all pose features to unit norm before learning? Sometimes helps.')    
    parser.add_argument('--embed_path', default='./data/poses/', type=str, help='path to pkl filename to store the embedded pose data. It will be saved as embed_path/data_train<split_num>.pkl and embed_path/data_test<split_num>.pkl')
    parser.add_argument('--verbose', action="store_true", help='echo some messages regarding status of the program.')
    parser.add_argument('--kernel', default='rbf', type=str, help='kernel to use for kgods. (currently, its linear/rbf/min/chisq. (default=rbf)')
    
    args = parser.parse_args()

    dir = "/workspace/data/objset2/calculated_features/GODS/result/KGODS2/sigma=" + str(args.sigma) + " eta=" + str(args.eta) + "/"

    result = np.zeros((7,1))
    tr_accuracy = np.zeros((7,1))
    for num in range(1):
        for anomalycls in range(0,7):
            gt = getf.get_ftr(anomalycls, objset=3)
            data_tr, label_tr, data_va, label_va, anomaly_object = gt.pcd_data_get()
            print(data_va.shape, label_va.shape, data_tr.shape, label_tr.shape)
            
            # data normalization. Make the embedded features unit norm.
            if not args.unnormalize:
                np.random.seed(seed)    
                data_tr = data_tr/data_tr.sum(1)[:,np.newaxis]
                data_va = data_va/data_va.sum(1)[:,np.newaxis]

            # Manifold setting
            d = data_tr.shape[1] # data dimensionality
            k = args.num_subspaces # number of subspaces.
            eta = args.eta #0.01  
            sigma = args.sigma
            num_pts = len(data_tr) # number of points.

            #one_kxk= np.ones((k, k), dtype='float')
            #one_kxn= np.ones((k, num_pts), dtype='float')
            one_nxk= np.ones((num_pts, k), dtype='float')
            X = data_tr
                    
            ##kokonikannsuuita
            # define the objective for optimization.
            def cost(M): # K, eta, one_nxk
                #print("cost beta", eta, "onenk", one_nxk)
                Y, Z = M[0], M[1]
                Y = Y*Y
                Z = Z*Z
                obj = 0.5*np.matmul(Y.transpose(), Y).sum() + \
                        np.trace(np.matmul(Y.transpose(), np.matmul(K, Z))) -\
                        eta*np.trace(np.matmul((Y-Z).transpose(), one_nxk)) 
                        
                obj += 0.1*np.linalg.norm(Y-Z)**2.0 # the lagrangian has (Y-Z)*1 = 0. We use a soft variant of that here.
                return obj
                
            # compute kernel for training set. 
            K = compute_kernel(X, X, sigma, num_pts, k, same=True)
            
            # Generalized Stiefel is a new manifold implementation, not available in pymanopt package. 
            manifold = Product((GeneralizedStiefel(num_pts, k, K), GeneralizedStiefel(num_pts,k, K))) 

            problem = Problem(manifold=manifold, cost=cost, verbosity=3) # problem setup.
            init_YZ = (np.random.randn(num_pts, k), np.random.randn(num_pts, k))
            if args.optim == 'cg':
                solver = ConjugateGradient(maxiter = args.max_iter) 
            elif args.optim == 'tr':
                solver = ConjugateGradient(maxiter = args.max_iter) 
            else:
                print('unknown solver. options are --optim=\'cg\' or --optim=\'tr\'')
                
            Xopt = solver.solve(problem) #, x = init_YZ) # solve the problem. This will use autograd for computing the gradients automatically.
            # compute the anomaly detection peformance. 
            print('neg norm=%f' %(np.linalg.norm(np.maximum(0,-Xopt[0]),'fro') + np.linalg.norm(np.maximum(0, -Xopt[1]), 'fro')))
            accu_tr, _, _, _ = calculate_accruacy(Xopt, data_tr, label_tr, eta, K)
            
            result[anomalycls, num] = calculate_AUC(Xopt, data_va, label_va, sigma, eta, K, k)    
            # print('split status')
            # print('-------------------')
            # #print('num train = %d \n num test = %d \n num test normal = num test abnormal = %d\n' % (data_tr.shape[0], data_va.shape[0], data_va_n.shape[0]-1))
            # print('-------------------')
            # print('Training accuracy is %.2f' % (accu_tr))
            # print('Test Evaluation:')    
            # print('Testing accuracy is %.2f' % (accu_va), '\nF1 score is %.2f ' % (F1), '\n Precision (abnormal samples) = %.2f ' % (precision[0]), '\n Recall (abnormal samples) =  %.2f' % (recall[0]))
    print(result)
    df = pd.DataFrame(result)
    df.to_csv(dir + str(args.num_subspaces) + "/auc_" + "_subdim_" + str(args.num_subspaces) + ".csv")
    print(result)
    df1 = pd.DataFrame(tr_accuracy)
    df1.to_csv(dir + str(args.num_subspaces) + "/train_accuracy_" + "_subdim_" + str(args.num_subspaces) + ".csv")