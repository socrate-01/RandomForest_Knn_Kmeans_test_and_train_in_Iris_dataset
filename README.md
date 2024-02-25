<h1> Testing and Training models in iris dataset using Random Forest; Nearest neighbors and Kmeans algorithms</h1>

<P>The main goal of this project is to understand how these algorithms works and to have a deeper understanding of supervised and unsupervised learning</P> <br>
<p> To run this project make sure to have python, and download the iris dataset. After that you'll be able to run each file.</p><br>

<h2>Random forest</h2>
<p>Random Forest is an ensemble technique used to improve the accuracy and robustness of machine learning models. The fundamental idea lies in building a large number of individual decision trees and combining their predictions to obtain a final result.
To better explain how random forest works, I've based myself on the official documentation in scikit learn.
It works as follows:
In the first instance, each tree in the forest is built from a random sample with replacement of the training set.
This means that at each iteration, a different subset of the data is used to build a tree.
When creating each node of a tree, the best split is sought either among all the input features, or among a random subset of these features. This randomization aims to reduce the variance of the forest estimators.
To do this, each tree "votes" for a class in the case of classification, or provides a value in the case of regression. The forest then chooses the majority class or the average of these values.
Randomization in tree construction reduces variance, meaning that the forest is less likely to over-fit the training data. By combining the predictions of multiple trees, the errors specific to each tree can cancel each other out, thus improving overall model performance.
</p>

<h2>Nearest Neighbors</h2>
<p>This Nearest Neighbor (k-NN) classification algorithm is a form of instance-based learning or non-generalized learning. Unlike algorithms that build a general internal model, k-NN simply stores instances from the training dataset.
When a new instance is to be classified, the algorithm identifies the k nearest training instances in the feature space (k is a user-specified parameter). The class of the unknown instance is determined by a majority vote among the classes of these neighbors.
Neighbors can be weighted in different ways when voting. By default, each neighbor has the same weight (uniform weight). However, it is possible to weight neighbors according to distance, for example. Several weighting methods are available, such as uniform weighting or inverse distance weighting.
The choice of the value of k is important and highly dependent on the data. A larger value of k may suppress the effects of noise, but may make the boundaries of the data more difficult to understand.
classification boundaries less distinct.
To better illustrate this algorithm, let's explore the example provided in scikit-learn.
The example uses scikit-learn's KNeighborsClassifier on the iris dataset to illustrate the impact of the weights parameter on the decision frontier. To do this, the iris data is loaded and divided into training and test sets. Then a k-NN classifier is created using a pipeline that includes a feature scaling step.
After this, two classifiers are fitted with different values of the weights parameter ("uniform" and "distance"). The decision boundaries of each classifier are displayed to observe the differences. These are visualized with the original dataset.
The example shows that the weights parameter has an impact on the decision frontier. When weights="uniform", all nearest neighbors have the same impact on the decision. On the other hand, when weights="distance", the weight assigned to each neighbor is proportional to the inverse of the distance between that neighbor and the query point.
Depending on the case, taking distance into account can improve the model, but this depends on the specific data. Visualizing decision boundaries can help to understand how the choice of parameter weights affects classification.
</p>

<h2>Kmeans</h2>
<p>The KMeans model is a data clustering algorithm that aims to separate samples into "k" groups of equal variance. To achieve this, it seeks to minimize a criterion called inertia or intra-cluster sum of squares. Inertia can be interpreted as a measure of the internal consistency of clusters. However, it has its limitations, as it assumes that clusters are convex and isotropic, which is not always the case. It is less responsive to elongated or irregularly shaped clusters.
The algorithm begins by initially selecting "k" centroids, usually by taking samples from the dataset. Then, each sample is assigned to the cluster whose centroid is closest, using Euclidean distance. The centroids of each cluster are recalculated by taking the average of the samples assigned to it. These allocation and updating steps are repeated until a stopping criterion is reached, often when the centroids no longer move significantly.
Inertia, although used as an evaluation criterion, has its drawbacks, particularly in terms of normalization, as it is not a normalized metric. In high-dimensional spaces, Euclidean distances tend to be inflated, leading to the "curse of dimensionality" problem. 
The use of dimension reduction techniques such as Principal Component Analysis (PCA) prior to clustering can alleviate this problem.
The KMeans algorithm is also known as Lloyd's algorithm. Its basic operation involves three steps: initialization of centroids, allocation of samples to clusters, and updating of centroids. Convergence is reached when centroids no longer move significantly.
KMeans benefits from OpenMP-based parallelism, processing small chunks of data in parallel for a reduced memory footprint. However, its convergence can depend on the initial choice of centroids, which often leads to running the algorithm several times with different initializations. A popular initialization method is k-means++, which places centroids at positions generally far apart to improve results.
In conclusion we can say that KMeans is a clustering algorithm that assigns samples to clusters while minimizing inertia, while dealing with limitations such as sensitivity to initialization and the need for multiple runs to mitigate these effects.

In exploring the algorithm, we can cite the example of "A demo of K-Means clustering on the handwritten digits data".
This example illustrates the application of the K-Means algorithm on a handwritten digits data set. The aim is to compare different initialization strategies in terms of execution time and quality of the results obtained.
The dataset used comprises handwritten digits from 0 to 9. In clustering,
the aim is to group images so that handwritten digits are similar within the same group.
To achieve this, three initialization approaches are compared in this example:
First we have k-means++, which shows that stochastic initialization is performed 4 times.
Then we have random, which shows that random initialization is also performed 4 times.
Finally, PCA-based, which shows that initialization is based on a PCA projection. This method is deterministic, and only one initialization is required.
The benchmark evaluates these approaches using various clustering quality metrics, such as homogeneity, completeness, V-measure, adjusted Rand index, and others. These metrics measure how well cluster labels fit the ground truth.
By visualizing the results on PCA-reduced data, the example demonstrates how clusters are distributed in two-dimensional space. 

Cluster centroids are marked with a white cross on the graph.
So we can say that through this example, we'll have a thorough understanding of the different K-Means initializations, their impact on clustering quality, and how the results can be evaluated using various metrics.

</p>

<h2>Some interstings links for more datasets</h2>
<P> https://archive.ics.uci.edu/</P><br>
https://guides.library.cmu.edu/machine-learning/datasets
