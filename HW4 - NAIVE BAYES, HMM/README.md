# Required Packages

 - NumPy


# How to run

## Naive Bayes

I have implemented the required functions in the **nb.py** file. For the testing part, I have created an additional file named **nb_util.py**. It contains these functions;

- filter_special_chars: To clean the data from special characters
- read_set: To read the dataset text files
- read_labels: To read the labels text files
- calculate_acc: To calculate the accuracy from scores and the actual labels.
- run_test: The main function, which is used to load files stored in the **nb_data** folder, calculate and print the test set accuracy

When **nb_util.py** is run, it will print the test set accuracy on the console. By changing the value of the **remove_special** parameter provided in the function call **run_test(remove_special=True)**, one can observe the accuracy difference between running with cleaned datasets and running with unprocessed datasets.



## HMM

I have implemented the required functions in the **hmm.py** file. There is no additional file like nb_util for hmm, since we are not expected to test it using an external dataset.
