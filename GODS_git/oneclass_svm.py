import numpy as np
from sklearn import svm
import get_features as getf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score

rst = []
result = np.zeros((7,1))
tr_accuracy = np.zeros((7,1))
for num in range(1):
    for anomalycls in range(0,7):
        auc = np.zeros((4,9))

        gt = getf.get_ftr(anomalycls)
        data_tr, label_tr, data_va, label_va, anomaly_object = gt.pcd_data_get()
        print(data_va.shape, label_va.shape, data_tr.shape, label_tr.shape)

        nu_list = [0.1, 0.001, 0.0001, 0.00001]
        gamma_list = [0.01, 0.1, 1, 10, 20, 30, 40, 50, 100]
        for i, nu in enumerate(nu_list):
            for j, gamma in enumerate(gamma_list):
                print("[nu = ", nu, "gamma", gamma, "]")
                # Train one-class SVM
                clf = svm.OneClassSVM(nu=nu, kernel="rbf", gamma=gamma)
                clf.fit(data_tr)

                # predict on current data
                prd_tr = clf.predict(data_tr)
                y_prd_tr = np.where(prd_tr == -1, 0, 1) # accuracy計算するときは正常が1で異常が0  元のラベルは最初からそう

                print("train data anomaly_num:",  len(y_prd_tr) - np.sum(y_prd_tr))
                print("train accuracy", accuracy_score(label_tr, y_prd_tr))

                # Predict on new data
                prediction = clf.predict(data_va)
                # Convert predictions to binary format (-1 for outliers, 1 for inliers) → (0 for outliers, 1 for inliers)
                y_pred_bin = np.where(prediction == -1, 0, 1) # accuracy計算するときは正常が1で異常が0  元のラベルは最初からそう
                print("test data anomaly_num:",  len(y_pred_bin) - np.sum(y_pred_bin))
                print("test accuracy", accuracy_score(label_va, y_pred_bin))


                label_va_hanten = [1 if n == 0 else 0 for n in label_va]

                scores = clf.decision_function(data_va)
                anomaly_score = 1 - ((scores - min(scores))/(max(scores) - min(scores)))

                auc_score = roc_auc_score(label_va_hanten, anomaly_score)
                print("auc", auc_score)

                auc[i, j] = auc_score


        rst.append(auc)

    for anomalycls in range(0,7):
        df = pd.DataFrame(rst[anomalycls])
        print(df)
            