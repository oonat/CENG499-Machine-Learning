# Required Packages

 - NumPy
 - Matplotlib
 - Scikit-learn
 - graphviz


# How to run

## DT

All the required functions are implemented in the **dt.py** file.

ID3 algorithm is implemented in the **id3.py** file. When the `python id3.py` command is run, by default, the program opens a pdf file that contains the visualization and the test set accuracy of the tree created by using **info\_gain** and without prepruning. Strategy and the prepruning option can be changed by editing the code in line 162.

root = id3(train_set, train_labels, 'info_gain', False)

For prepruning just change False to True.




## SVM

### Task 1

Task 1 is implemented in a file named **svm_task1.py**. By running the `python svm_task1.py` command, one can create necessary plots for classifiers. This command will create a plot for each classifier with a name;

"svm1_{c value}.png"


### Task 2

Task 2 is implemented in a file named **svm_task2.py**. By running the `python svm_task2.py` command, one can create necessary plots for classifiers. This command will create a plot for each classifier with a name;

"svm2_{kernel name}.png"


### Task 3

Task 3 is implemented in a file named **svm_task3.py**. When the `python svm_task3.py` command is run, the validation accuracy for each parameter configuration, the best configuration, and the test set accuracy for the best configuration will be printed in the terminal.

I tried 18 different combinations for these values {'kernel': ('linear', 'rbf', 'poly'), 'C': [0.1, 1, 10], 'gamma': [0.001, 0.01]}


### Task 4

Task 4 is implemented in a file named **svm_task4.py**. When the `python svm_task4.py` command is run, the test set accuracy and confusion matrix for each dataset will be printed in the terminal. Example result;

--------IMBALANCED--------
Accuracy: 0.95
Confusion matrix: [[  0  50]
 [  0 950]] 

--------OVERSAMPLED--------
Accuracy: 0.951
Confusion matrix: [[ 12  38]
 [ 11 939]] 

--------UNDERSAMPLED--------
Accuracy: 0.798
Confusion matrix: [[ 33  17]
 [185 765]] 

--------CLASS WEIGHT BALANCED--------
Accuracy: 0.936
Confusion matrix: [[ 16  34]
 [ 30 920]]
