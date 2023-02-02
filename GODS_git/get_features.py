import numpy as np
import pandas as pd
import sys

    
class get_ftr():
    def __init__(self, Anomaly_class: int):
        self.names = ['bookshelf', 'laptop', 'knife', 'train', 'motorbike', 'guitar', 'faucet']
        self.folder_names_test = ["180", "184", "170", "156", "134", "320", "298"]

        self.Anomaly_class = Anomaly_class
        self.CLASS_NUM = 7
        self.args = sys.argv
        self.Anomaly_object = self.names[self.Anomaly_class]                                 # args[1]
        #Anomaly_object = 'airplane'                                            # ←ここでobject指定すること!
        self.test_folder = "c_epoc_299_data" + self.folder_names_test[self.Anomaly_class]    #args[2]
        #test_folder = 'c_epoc_299_data576' #←例
        self.test_way = "test" #args[3]
        #val test
        self.dic_folder = "c_epoc_299_data2521" #args[4]
        #dic_folder = 'c_epoc_299_data2521' #←例


        self.dictionary_dir =  '../data/objset2/calculated_features/modelAE_' + self.Anomaly_object + '/both_features/' + self.dic_folder + '/'
        self.test_dir = '../data/objset2/calculated_features/modelAE_' + self.Anomaly_object + '/both_features/' + self.test_folder + '/'

        self.dic_names = {'317': 'bookshelf', '322': 'laptop', '297':'knife', '272':'train', '236':'motorbike', '557': 'guitar', '520': 'faucet'}
        if self.test_way == 'test':
            self.dic_names_test = {'90': 'bookshelf', '92': 'laptop', '85':'knife', '78':'train', '67':'motorbike', '160': 'guitar', '149': 'faucet'}
        else:
            self.dic_names_test = {'45': 'bookshelf', '46': 'laptop', '42':'knife', '39':'train', '34':'motorbike', '80': 'guitar', '75': 'faucet'} # val用


    def get_features_and_sort_onlymu(self, dir):
        names = pd.read_csv(dir + 'name.csv')
        mu = pd.read_csv(dir + 'mu.csv')
        
        names = names.iloc[:, 1:].values.ravel()
        mu = mu.iloc[:, 1:].values
        
        clsnm = np.zeros(self.CLASS_NUM) # 各クラスのデータ数
        m = np.zeros((mu.shape))
        y = np.zeros(mu.shape[0])

        for cnt in range(self.CLASS_NUM):
            idx = [i for i, x in enumerate(names.tolist()) if x == cnt] # 特定のクラスのindexを抽出
            m[int(np.sum(clsnm)) : (int(np.sum(clsnm) + len(idx))), :] = mu[idx, :]
            y[int(np.sum(clsnm)) : (int(np.sum(clsnm) + len(idx)))] = cnt
            clsnm[cnt] = int(len(idx))

        clsnm = np.array(clsnm, dtype=int)
        num = np.zeros(self.CLASS_NUM + 1)
        for i in range(self.CLASS_NUM + 1):
            num[i] = int(np.sum(clsnm[:i]))
        num = np.array(num, dtype=int)

        return m, clsnm, num, y   # feature, mu, var, y 

    def set_anomaly_labels(self, ftr, clsnm, num, anomaly_obj):

        anomaly_labels = np.ones(len(ftr))

        anomaly_num = [k for k, v in self.dic_names_test.items() if v == anomaly_obj]
        anomaly_num = int(anomaly_num[0])
        
        for i in range(len(clsnm)):
            if clsnm[i] == anomaly_num:
                anomaly_labels[num[i]: num[i+1]] = 0 # GODSと合わせてanomaly=0 

        return anomaly_labels


    def exclude_anomaly_obj(self, ftr, clsnm, num, anomaly_obj):

        # object_name → data_size取り出し
        anomaly_num = [k for k, v in self.dic_names.items() if v == anomaly_obj]
        anomaly_num = int(anomaly_num[0])

        # 
        ftre = np.zeros((ftr.shape[0] - anomaly_num, ftr.shape[1]))
        for i in range(len(num) - 1):
            if (num[i + 1] - num[i]) == anomaly_num:
                ftre = np.vstack((ftr[:num[i]], ftr[num[i+1]:]))
                clsnm = np.delete(clsnm, i)
        
        num = np.zeros(self.CLASS_NUM)
        for i in range(self.CLASS_NUM):
            num[i] = int(np.sum(clsnm[:i]))
        num = np.array(num, dtype=int)

        return ftre, clsnm, num


    # train test data get
    def pcd_data_get(self):

        # テスト開始
        sub_dim = 80
        train_ftr, clsnm, num, _ = self.get_features_and_sort_onlymu(self.dictionary_dir)

        print('Anomaly_Class:', self.Anomaly_object)

        #TrainData
        print('TrainData: clsnm', clsnm, 'num', num, '\n', [self.dic_names[str(clsnm[k])] for k in range(self.CLASS_NUM)])

        # TrainDataからAnomalyClassを排除
        ftr, clsnm, num = self.exclude_anomaly_obj(train_ftr, clsnm, num, self.Anomaly_object)
        train_labels = np.ones(len(ftr)) 
        subspace_class_num = self.CLASS_NUM - 1

        print('\n<TrainData (without AnomalyClass) & Labels>')
        for i in range(subspace_class_num):
            print(self.dic_names[str(clsnm[i])], clsnm[i], '\t', num[i], ':', num[i+1], '\t', int(train_labels[num[i]]))

        # テストデータ
        test_ftr, test_clsnm, test_num, y = self.get_features_and_sort_onlymu(self.test_dir)
        test_labels = self.set_anomaly_labels(test_ftr, test_clsnm, test_num, self.Anomaly_object)

        print('\n<Test_dataname & Test_datanum & test_Labels>:')
        for i in range(self.CLASS_NUM):
            print(self.names[i], test_clsnm[i],'\t', test_num[i], ':', test_num[i+1], '\t', int(test_labels[test_num[i]]))

        return(ftr, train_labels, test_ftr, test_labels, self.Anomaly_object)

    # if Anomaly_object in names:

    #     train_x, train_labels, test_x, test_labels = pcd_data_get()
    #     print('train:', train_x.shape, train_labels, '\ntest:', test_x.shape, test_labels)
    # else:
    #     print('command line input error')
