%% Load and Vectorize Data 

train_im = loadMNISTImages('../data/mnist/train-images-idx3-ubyte');
train_labels = loadMNISTLabels('../data/mnist/train-labels-idx1-ubyte')';

test_im = loadMNISTImages('../data/mnist/t10k-images-idx3-ubyte');
test_labels = loadMNISTLabels('../data/mnist/t10k-labels-idx1-ubyte')';

% Shift labels by 1 to allow indexing
train_labels = train_labels + 1;
test_labels = test_labels + 1;

% Reduced sample sizes for testing
reduced_size = 10000;
if reduced_size > 0
	train_im = train_im(:,1:reduced_size);
	train_labels = train_labels(1:reduced_size);
	
	test_im = test_im(:,1:reduced_size);
	test_labels = test_labels(1:reduced_size);
end

%% Dimensionality Reduction

dprime = 400;

% PCA

tic
principle_coeffs = myPca(train_im);
pca_time = toc

principle_coeffs = principle_coeffs(1:dprime,:);

train_pca = principle_coeffs * train_im;
test_pca = principle_coeffs * test_im;

% LDA
tic
lda_coeffs = myLda(train_im, train_labels);
lda_time = toc

lda_coeffs = lda_coeffs(1:dprime,:);

train_lda = lda_coeffs * train_im;
test_lda = lda_coeffs * test_im;

%% Classification
% Baye's Classification

tic
bayes_labels = myBayes(train_im, train_labels, test_im);
bayes_time = toc

tic
bayes_labels_pca = myBayes(train_pca, train_labels, test_pca);
bayes_pca_time = toc

tic
bayes_labels_lda = myBayes(train_lda, train_labels, test_lda);
bayes_lda_time = toc

% kNN Classification

tic
knn_labels = myKnn(1, train_im, train_labels, test_im);
knn_time = toc

tic
knn_labels_pca = myKnn(1, train_pca, train_labels, test_pca);
knn_pca_time = toc

tic
knn_labels_lda = myKnn(1, train_lda, train_labels, test_lda);
knn_lda_time = toc

%% Error Rates

bayes_error = bayes_labels - test_labels;
bayes_error = bayes_error ~= 0;
bayes_error = sum(bayes_error)

bayes_pca_error = bayes_labels_pca - test_labels;
bayes_pca_error = bayes_pca_error ~= 0;
bayes_pca_error = sum(bayes_pca_error)

bayes_lda_error = bayes_labels_lda - test_labels;
bayes_lda_error = bayes_lda_error ~= 0;
bayes_lda_error = sum(bayes_lda_error)

knn_error = knn_labels - test_labels;
knn_error = knn_error ~= 0;
knn_error = sum(knn_error)

knn_pca_error = knn_labels_pca - test_labels;
knn_pca_error = knn_pca_error ~= 0;
knn_pca_error = sum(knn_pca_error)

knn_lda_error = knn_labels_lda - test_labels;
knn_lda_error = knn_lda_error ~= 0;
knn_lda_error = sum(knn_lda_error)

