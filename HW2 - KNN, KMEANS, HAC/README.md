## Required Packages

 - NumPy
 - Matplotlib

## How to run

### K-Nearest Neighbors
All the required implementations are placed in knn.py. However, to plot the graphs and calculate the test set accuracies, I created an additional file named knn_util.py. One can draw the plots and calculate the test set accuracies by using the `python knn_util.py` command. 

This command will create two png files named "KNN_L1_PLOT.png" and "KNN_L2_PLOT.png", which contain required plots for Manhattan Distance and Euclidean Distance, respectively. 

In addition, the command will print the best k values and the corresponding test set accuracies for L1 and L2 distance metrics to the terminal.

### K-Means
All the required implementations are placed in kmeans.py (including the additional initialize_cluster_centers function). However, to plot the graphs, I created an additional file named kmeans_util.py. One can draw the plots using the `python kmeans_util.py` command. 

This command will create k versus objective function and cluster colorization plots for each dataset. 

Name for k versus objective function plots;
"Dataset{id}\_kmeans_k_versus_obj.png"

Name for cluster colorization;
"Dataset{id}\_cluster_plot_k_{k value}.png"

### HAC
All the required implementations are placed in hac.py. However, to plot the graphs, I created an additional file named hac_util.py. One can draw the plots using the `python hac_util.py` command. 

This command will create cluster colorization plots for each dataset and criterion. 

Name for cluster colorization;
"Dataset{id}\_hac_ {criterion name}.png"
