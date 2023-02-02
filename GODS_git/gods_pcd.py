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

""" Generalized One-Class Discriminative Subspaces (GODS) for anomaly detection. 
    Implementation in Python using PyManOpt. 
    implemented by Anoop Cherian and Jue Wang.
"""
from re import T
import autograd.numpy as np
from pymanopt import Problem
from pymanopt.solvers import ConjugateGradient, TrustRegions
from pymanopt.manifolds import Product, Stiefel, Euclidean, Sphere, Oblique
import argparse
import os
import random
import pickle
import pdb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix, auc, roc_curve
from sklearn.preprocessing import normalize
import pymanopt
import get_features as getf
import matplotlib
import pandas as pd

def init_subspaces(X, num_subspaces, eta):     
    if num_subspaces > 1:
        w1, w2, b1, b2 = init_stiefel(X, num_subspaces, eta)
        #w1,_,_ = np.linalg.svd(np.transpose(X[idx[:num_subspaces], :]), full_matrices=False)
        #w2,_,_ = np.linalg.svd(np.transpose(X[idx[-num_subspaces:], :]), full_matrices=False)        
    else:
        dist = np.sqrt(np.square(X).sum(1))
        idx = np.argsort(dist)
        w1 = X[idx[0], :]
        w2 = X[idx[-1], :]
        w1 /= (np.linalg.norm(w1) + 1e-10)
        w2 /= (np.linalg.norm(w2) + 1e-10)
    
    return (w1, w2, np.zeros((1, num_subspaces),dtype='float'), np.zeros((1, num_subspaces), dtype='float'))

def init_stiefel(X, num_subspaces, eta):
    manifold = Product((Stiefel(d, k), Euclidean(1, k)))
    data = np.transpose(X) 
#    @pymanopt.function.Autograd
    def cost_lower(M):
        w, b = np.transpose(M[0]), np.transpose(M[1]) # the subspaces.
        ww = np.dot(w, data) + b  * np.ones((X.shape[0],))
        upper = np.maximum(0,   np.add(eta, np.max(ww, axis=0))) 
        obj = np.sum(np.square(upper)) + np.sum(np.square(b)) 
        return obj
    
    #@pymanopt.function.Autograd
    def cost_upper(M):
        w, b = np.transpose(M[0]), np.transpose(M[1]) # the subspaces.
        ww = np.dot(w, data) + b * np.ones((X.shape[0],))
        lower = np.maximum(0,   np.add(eta, -np.min(ww, axis=0))) 
        obj = np.sum(np.square(lower)) + np.sum(np.square(b))
        return obj

    solver = ConjugateGradient(maxiter = 100)
    problem_upper = Problem(manifold=manifold, cost=cost_upper, verbosity=int(args.verbose)*3)
    w2,b2 = solver.solve(problem_upper) 
    problem_lower = Problem(manifold=manifold, cost=cost_lower, verbosity=int(args.verbose)*3)
    w1, b1 = solver.solve(problem_lower)
    return w1, w2, b1, b2

seed = 123
print('seed=%d'%(seed))
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='GODS: pose anomaly detection.')
    parser.add_argument('--split_num', default=0, type=int, help='in case of cross-validation, specfiy the cross-validation number (1/2/3/4)')    
    parser.add_argument('--num_subspaces', default=2, type=int, help='number of hyperplanes in each of GODS subspaces. Default=2')    
    parser.add_argument('--eta', default=0.01, type=float, help='eta, controlling how far are the predictions from the hyperplanes. Default=0.01')        
    parser.add_argument('--L', default=0.001, type=float, help='lambda, regularization cost for the distance between subspaces. Default=0.001') 
    parser.add_argument('--max_iter', default=100, type=int, help='max-number of Riemannian optimization iterations. Default=100')    
    parser.add_argument('--optim', default='cg', type=str, help='Optimizer to use. conjugate-gradient (cg) / trust-regions (tr). Default=cg')    
    parser.add_argument('--thresh', default=0., type=float, help='threshold for deciding if the given data point is within the two subspaces. Default=0')    
    parser.add_argument('--unnormalize', action='store_true', help='Should you normalize all pose features to unit norm before learning? Sometimes helps.')    
    parser.add_argument('--embed_path', default='./data/poses/', type=str, help='path to pkl filename to store the embedded pose data. It will be saved as embed_path/data_train<split_num>.pkl and embed_path/data_test<split_num>.pkl')
    parser.add_argument('--verbose', action="store_true", help='echo some messages regarding status of the program.')
    args = parser.parse_args()

    # Data prepare
    # pdb.set_trace()
    # with open(os.path.join(args.embed_path, 'data_train' + str(args.split_num) + '.pkl'), 'rb') as handle:
    #     data_tr = pickle.load(handle, encoding='latin1')
    # with open(os.path.join(args.embed_path, 'data_test' + str(args.split_num) + '.pkl'), 'rb') as handle:
    #     data_va_n = pickle.load(handle, encoding='latin1')
    
    # # select part of the training set as normal data for testing. We will use all of the negative data for testing. 
    # random.seed(seed)
    # index = random.sample(range(0, data_tr.shape[0]), data_va_n.shape[0] - 2)    
    # data_va_p = np.array([data_tr[i, :] for i in range(data_tr.shape[0]) if i in index])
    # data_tr = np.array([data_tr[i, :] for i in range(data_tr.shape[0]) if i not in index])

    # data_va = np.concatenate((data_va_n, data_va_p), axis=0)
    # label_va = np.ones((data_va.shape[0],), dtype=int)
    # label_va[:data_va.shape[0] // 2] = 0
    # label_tr = np.ones((data_tr.shape[0],), dtype=int)

    # print(data_va.shape, label_va.shape, data_tr.shape, label_tr.shape)
    

    # define the objective for optimization.
    #@pymanopt.function.Autograd
    def cost(M):
        w1, w2, b1, b2 = np.transpose(M[0]), np.transpose(M[1]), np.transpose(M[2]), np.transpose(M[3]) # the subspaces.
        data = np.transpose(data_tr) 
        
        ww1 = np.dot(w1, data) + b1  * np.ones((data_tr.shape[0],))
        ww2 = np.dot(w2, data) + b2 * np.ones((data_tr.shape[0],))
        
        lower = np.maximum(0,   np.add(eta, -np.min(ww1, axis=0))) # 要素ごとに最大の値を取り出す
        upper = np.maximum(0,   np.add(eta, np.max(ww2, axis=0))) 

        obj = np.sum(np.square(lower)) + np.sum(np.square(upper)) + Lambda*(np.sum(np.square(ww1)) + np.sum(np.square(ww2)))
        
        return obj

    # compute prediction accuracy.
    def calculate_accruacy(opt, data, label_gt):
        label = []
        w1, w2, b1, b2 = np.transpose(opt[0]), np.transpose(opt[1]), np.transpose(opt[2]), np.transpose(opt[3])
        for i in range(len(data)):
            item = np.transpose(data[i, :])
            ww1 = np.expand_dims(np.matmul(w1, item), axis=1) + b1
            ww2 = np.expand_dims(np.matmul(w2, item), axis=1) + b2
            
            # we use exact thresholds for judging accuracy. That is, if the pose falls between the two subspaces, we take that as normal class, else abnormal.
            if np.min(ww1) > args.thresh and np.max(ww2) < args.thresh:
                label.append(1)
            else:
                label.append(0)
        #print("predicted_label", label)
        accuracy = accuracy_score(label_gt, label)
        F1 = f1_score(label_gt, label)        
        precision, recall, fbeta_score, support = precision_recall_fscore_support(label_gt, label)
        return accuracy, F1, precision, recall


    # compute prediction accuracy.
    def calculate_accruacy_thresh(opt, data, label_gt):
        ftr_roc_auc = 0
        tpr = []
        fpr = []
        cls0 = []
        cls1 = []
        thresh_arr = []
        for j in range(-120, 120, 1):
            thresh = j*0.001
            thresh_arr.append(round(thresh,4))
            label = []
            w1, w2, b1, b2 = np.transpose(opt[0]), np.transpose(opt[1]), np.transpose(opt[2]), np.transpose(opt[3])
            for i in range(len(data)):
                item = np.transpose(data[i, :])
                ww1 = np.expand_dims(np.matmul(w1, item), axis=1) + b1  # distanceはww1/||w1||だが各基底は正規直交基底なのでノルム1なのでこのままでよし
                ww2 = np.expand_dims(np.matmul(w2, item), axis=1) + b2
                
                # we use exact thresholds for judging accuracy. That is, if the pose falls between the two subspaces, we take that as normal class, else abnormal.
                if np.min(ww1) > thresh and np.max(ww2) < -(thresh):
                    label.append(1)
                else:
                    label.append(0)
            #print("thresh", args.thresh)

            # 0と1を反転
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

            # 以下は元の論理
            #print(confusion_matrix(label_gt, label))
            accuracy = accuracy_score(label_gt, label)
            F1 = f1_score(label_gt, label)        
            precision, recall, fbeta_score, support = precision_recall_fscore_support(label_gt, label)
            #print('Test Evaluation:')    
            #print('Testing accuracy is %.2f' % (accuracy), '\nF1 score is %.2f ' % (F1), '\n Precision (abnormal samples) = %.2f ' % (precision[0]), '\n Recall (abnormal samples) =  %.2f' % (recall[0]))

        if (min(tpr) == 0 and min(fpr) == 0 and max(cls1) == len(label_gt) and max(cls0) == len(label_gt)):
            tpr = tpr / max(tpr)
            fpr = fpr / max(fpr)
            # print("tp",tpr)
            # print("fp", fpr)
            ftr_roc_auc = auc(fpr, tpr)
            print(ftr_roc_auc)
            print("thresh", thresh_arr)
            print("num of cls0", cls0)
            print("num of cls1", cls1)

            plt.figure(figsize=(5,5))
            plt.plot(fpr, tpr)
            plt.xlabel('FPR: False Positive Rate', fontsize = 13)
            plt.ylabel('TPR: True Positive Rate', fontsize = 13)
            plt.grid()
            plt.savefig("/workspace/result/GODS/" + str(k) + "/auc_" + anomaly_object + "_subdim_" + str(k) + "_auc_" + str(round(ftr_roc_auc,3)) + ".png")
        
        
        
        return ftr_roc_auc #accuracy, F1, precision, recall

    result = np.zeros((7,1))
    tr_accuracy = np.zeros((7,1))
    for num in range(1):
        for anomalycls in range(0,7):
            gt = getf.get_ftr(anomalycls)
            data_tr, label_tr, data_va, label_va, anomaly_object = gt.pcd_data_get()
            print(data_va.shape, label_va.shape, data_tr.shape, label_tr.shape)

            # data normalization. Make the embedded features unit norm.
            if not args.unnormalize:
                np.random.seed(seed)    
                data_tr = normalize(data_tr, axis=1, norm='l2')
                data_va = normalize(data_va, axis=1, norm='l2')

            # Manifold setting
            d = data_tr.shape[1] # data dimensionality
            k = args.num_subspaces # number of subspaces.
            eta = args.eta #0.01  
            Lambda = args.L #0.001 
            num_pts = len(data_tr) # number of points.


            # setup the manopt framework.
            if k > 1:
                manifold = Product((Stiefel(d, k), Stiefel(d, k), Euclidean(1, k), Euclidean(1, k))) # product manifold of two Stiefels and their biases.
            else:
                # if we use only one subspace, then its better to use Sphere manifold for efficiency.
                manifold = Product((Sphere(d), Sphere(d), Euclidean(1, k), Euclidean(1, k))) # product manifold of two Stiefels and their biases.
                
            problem = Problem(manifold=manifold, cost=cost, verbosity=int(args.verbose)*3) # problem setup.
            if args.optim == 'cg':
                solver = ConjugateGradient(maxiter = args.max_iter) # we use Riemannian Conjugate gradient. Another option is to use TrustRegions. 
            elif args.optim == 'tr':
                solver = ConjugateGradient(maxiter = args.max_iter) # we use Riemannian Conjugate gradient. Another option is to use TrustRegions. 
            else:
                print('unknown solver. options are --optim=\'cg\' or --optim=\'tr\'')
                
            init_X = init_subspaces(data_tr, k, eta)
            Xopt = solver.solve(problem, x = init_X) # solve the problem. This will use autograd for computing the gradients automatically.

            
            # compute the anomaly detection peformance.
            print('split status')
            print('-------------------')
            
            accu_tr, _, _, _ = calculate_accruacy(Xopt, data_tr, label_tr)
            print('Training accuracy is %.2f' % (accu_tr))
            tr_accuracy[anomalycls, num] = accu_tr
            #accu_va, F1, precision, recall = 
            re = calculate_accruacy_thresh(Xopt, data_va, label_va)    
            result[anomalycls, num] = re
    
            #print('num train = %d \n num test = %d \n num test normal = num test abnormal = %d\n' % (data_tr.shape[0], data_va.shape[0], data_va_n.shape[0]-1))
            #print('Test Evaluation:')    
            #print('Testing accuracy is %.2f' % (accu_va), '\nF1 score is %.2f ' % (F1), '\n Precision (abnormal samples) = %.2f ' % (precision[0]), '\n Recall (abnormal samples) =  %.2f' % (recall[0]))
    print(result)
    df = pd.DataFrame(result)
    df.to_csv("/workspace/result/GODS/" + str(args.num_subspaces) + "/auc_" + "_subdim_" + str(args.num_subspaces) + ".csv")
    print(result)
    df1 = pd.DataFrame(tr_accuracy)
    df1.to_csv("/workspace/result/GODS/" + str(args.num_subspaces) + "/train_accuracy_" + "_subdim_" + str(args.num_subspaces) + ".csv")
    