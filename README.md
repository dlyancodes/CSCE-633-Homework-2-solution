# CSCE-633-Homework-2-solution

Download Here: [CSCE 633 Homework 2 solution](https://jarviscodinghub.com/assignment/csce-633-homework-2-solution/)

For Custom/Original Work email jarviscodinghub@gmail.com/whatsapp +1(541)423-7793

Question 1: Decision Tree
(a) Suppose we have 80 observations representing in the following tabulated data with binary
features, such as temperature, humidity, and sky condition, and we observe the occurrence of
the rainy day, denoted by total rainy days per total observations (#Rainy/#Observations) for
each combination of features. Using this data, we want to grow a decision tree which maximizes
information gain, to predict the future occurrence rainy days. Please provide (i) the intermediate
computations, (ii) the predictor variable/feature that you select for the split in each node of
the tree based on information gain criteria, and (iii) draw the resulting decision tree.
(b) In training decision trees, the ultimate goal is to minimize the classification error. However,
the classifaction error is not a smooth function; thus, several surrogate loss functions have been
proposed. Two of the most common loss functions are the Gini index and Cross-entropy. Prove
that, for any discrete probability distribution p with K classes, the value of the Gini index is
less than or equal to the corresponding value of the entropy. This implies that the Gini index
is a better approximation of the misclassification error.
Definitions: For a K-valued discrete random variable with probability mass function pi
, i =
1, . . . , K the Gini index is defined as: φG(p1, . . . , pK) = PK
k=1 pk(1 − pk) and the entropy is
defined as φE(p1, . . . , pK) = PK
k=1 pklogpk.
(c) Classifying benign vs malignant tumors: We would like to classify if a tumor is
benign or malign based on its attributes. We use data from the Breast Cancer Wisconsin Data
Set of the UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/
breast+cancer+wisconsin+(original).
1
Inside “Homework 2” folder on Piazza you can find one file containing the data (named “hw2
question1.csv”) for our experiments. The rows of these files refer to the data samples, while the
columns denote the features (columns 1-9) and the outcome variable (column 10), as described
bellow:
1. Clump Thickness: discrete values {1, 10}
2. Uniformity of Cell Size: discrete values {1, 10}
3. Uniformity of Cell Shape: discrete values {1, 10}
4. Marginal Adhesion: discrete values {1, 10}
5. Single Epithelial Cell Size: discrete values {1, 10}
6. Bare Nuclei: discrete values {1, 10}
7. Bland Chromatin: discrete values {1, 10}
8. Normal Nucleoli: discrete values {1, 10}
9. Mitoses: discrete values {1, 10}
10. Class: 2 for benign, 4 for malignant (this is the outcome variable)
(c.i) Compute the number of samples belonging to the benign and the number of samples
belonging to the malignant case. What do you observe? Are the two classes equally represented
in the data? Separate the data into a train (2/3 of the data) and a test (1/3 of the data) set.
Make sure that both classes are represented with the same proportion in both sets.
(c.ii) Implement two decision trees using the training samples. The splitting criterion for the
first one should be the entropy, while for the second one should be the gini index. Plot the
accuracy on the train and test data while the number of nodes in the tree increases for both
splitting criteria. Do you observe any differences in practice?
(c.ii) Bonus: Implement pre-pruning using a lower threshold on the values of the splitting
criterion for each branch. Experiment with different thresholds and report results both in the
train and test set.
Question 2: Kernel Ridge Regression
In this problem, we will derive kernel ridge regression, a nonlinear extension of linear ridge
regression. Given a set of training data {(x1, y1), . . . ,(xN, yN )} where xn ∈ RD, linear ridge
regression learns the weight vector w (assuming the bias term is absorbed into w) by optimizing
the following objective function:
J(w) = (y − Xw)
T
(y − Xw) + λkwk
2
2
where λ is the regularization coefficient.
Assume that we apply a nonlinear feature mapping to each sample xn → Φi = φ(xn) ∈ RT
,
where T  D. Define Φ ∈ RN×T as a matrix containing all Φn.
(a) Express the above criterion function in terms of the non-linear transform φ and show that
the weights w∗
that minimize the criterion function can be written as
w∗ = ΦT
(ΦΦT + λIN )
−1y
2
where y = [y1, . . . , yN ]
T and X =



− x
T
1 −
.
.
.
− x
T
N −



Hint: You may use the following identity for matrices. For any matrix P ∈ Rp×p
, B ∈ Rq×p
,
R ∈ Rq×q and assume the matrix inversion is valid, we have
(P
−1 + B
T R−1B)
−1B
T R−1 = PBT
(BPBT + R)
−1
(b) Given a testing sample φ(x), show that the prediction y = w∗T φ(x) can be written as:
y = y
T
(K + λIN )
−1κ(x)
where K ∈ RN×N is a kernel matrix defined as Kij = Φi
T Φj , κ(x) ∈ RN is a vector with nth
element (κ(x))n = φ
T
(xn)φ(x). Now you can see that y only depends on the dot product (or
kernel value) of Φi.
(c) Bonus: Compare the computational complexity between linear ridge regression and kernel
ridge regression.
Question 3: Support Vector Machines
We will use the Phishing Websites Data Set from UCI’s machine learning data repository:
https://archive.ics.uci.edu/ml/datasets/Phishing+Websites. The dataset is for a binary classification problem to detect phishing websites.
Inside “Homework 2” folder on Piazza you can find a file containing the data (named “hw2
question3.csv”) for our experiments. The rows of these files refer to the data samples, while the
columns denote the features (columns 1-30) and the binary outcome variable (column 31).
(a) Data pre-processing: All the features in the datasets are categorical. You need to
preprocess the training and test data to make features with multiple values to features taking
values only zero or one. If a feature fi have value {−1, 0, 1}, we create three new features fi,−1,
fi,0, and fi,1. Only one of them can have value 1 and fi,x = 1 if and only if fi = x. For example,
we transform the original feature with value 1 into [0, 0, 1]. In the given dataset, the features
2, 7, 8, 14, 15, 16, 26, 29 (index starting from 1) take three different values {−1, 0, 1}. You
need to transform each above feature into three 0/1 features. For all the following experiments
randomly separate the data into train (2/3) and test (1/3) set.
(b) Use linear SVM in LIBSVM: LIBSVM is widely used toolbox for SVM and has Matlab
interface. Download LIBSVM from https://www.csie.ntu.edu.tw/~cjlin/libsvm/ and install it according to the README file provided with the download. Experiment with different
values of misclassification cost C, applying 3-fold cross validation on the train set and reporting
the cross validation accuracy and average training time. Report the results on the test set using
the best C that was found through cross-validation.
(c) Use kernel SVM in LIBSVM: LIBSVM supports a number of kernel types. Here you
need to experiment with the polynomial kernel and RBF (Radial Basis Function) kernel and
their different parameters. Based on the cross validation results of Polynomial and RBF kernel,
which kernel type and kernel parameters will you choose?
(d) Bonus: Implement linear SVM: Implement the training and testing parts of a linear
support vector machine. In your implementation, you can use publicly available quadratic
programming functions (e.g. quadprog in Matlab) to solve the dual quadratic problem.

