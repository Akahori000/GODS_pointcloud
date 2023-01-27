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


How to run the code:

1. Run the following command for training the kgods model
python kgods.py --embed_path ./data/embed/ --split_num 1 --eta 0.0001 --num_subspaces 1 --sigma 0.01  

2. Run the following command for training and testing using the improved gods model. 
python gods.py --embed_path ./data/embed/ --split_num 1 --eta 0.01 --max_iter 1000 --num_subspaces 5 --L 0.1 --unnormalize

You should get the following results if everything goes well. 

Folder details:
./multclass/tmp has preprocessed pose data extracted using openpose
./data/gt: has train/val splits
./data/embed: is where bow embedded data (from bow_embed_pose.py) is stored.

**** Note ****
You need to install PyManOpt to run the implementations (gods.py and kgods.py). The kgods.py code uses a custom implementation of the 
GenerarlizedStiefel manifold, which is not available publicly, but is implemented by us. So you need to install the attached 0.2.4 version 
of the pymanopt code, to run our software. 

Other python dependencies are provided in python_dependencies.txt

For questions, please email: cherian@merl.com

training logs #####################################################

# output on gods.py
python gods.py --embed_path ./data/embed/ --split_num 1 --eta 0.01 --max_iter 1000 --num_subspaces 5 --L 0.1 --unnormalize

(python3.7) bash-3.2$ python gods.py --embed_path ./data/embed/ --split_num 1 --eta 0.01 --max_iter 1000 --num_subspaces 5 --L 0.1 --unnormalize
seed=123
/Users/cherian/projects/tmp/python3.7/lib/python3.7/site-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
split status
-------------------
num train = 354 
 num test = 58 
 num test normal = num test abnormal = 29

-------------------
Training accuracy is 0.97
Test Evaluation:
Testing accuracy is 0.74 
F1 score is 0.75  
 Precision (abnormal samples) = 0.77  
 Recall (abnormal samples) =  0.69



# output on kgods.py
$ python kgods.py --embed_path ./data/embed/ --split_num 1 --eta 0.0001 --num_subspaces 1 --sigma 0.01
mean of the kernel=0.007794
Compiling cost function...
Computing gradient of cost function...
Optimizing...
 iter		   cost val	    grad. norm
    0	+1.3849796054260045e-02	2.64810869e+01
    1	+1.3849796054260045e-02	2.64810869e+01
Terminated - min stepsize reached after 2 iterations, 0.01 seconds.

neg norm=1.443132
mean of the kernel=0.007794
/Users/cherian/projects/tmp/python3.7/lib/python3.7/site-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
mean of the kernel=0.002845
split status
-------------------
num train = 354 
 num test = 58 
 num test normal = num test abnormal = 29

-------------------
Training accuracy is 0.99
Test Evaluation:
Testing accuracy is 0.81 
F1 score is 0.80  
 Precision (abnormal samples) = 0.78  
 Recall (abnormal samples) =  0.86

