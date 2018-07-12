%% Load and Vectorize Data
train_size = 10000;
test_size = 10000;
[train_im, train_labels, test_im, test_labels] = load_mnist(train_size, test_size);

%% Dimensionality Reduction

dprime = 400;

% PCA

tic
principle_coeffs = my_pca(train_im);
pca_time = toc

principle_coeffs = principle_coeffs(1:dprime,:);

train_pca = principle_coeffs * train_im;
test_pca = principle_coeffs * test_im;

% LDA
tic
mda_coeffs = mda(train_im, train_labels, .95);
mda_time = toc

train_mda = mda_coeffs * train_im;
test_mda = mda_coeffs * test_im;

% kNN Classification

tic
knn_labels = knn(1, train_im, train_labels, test_im);
knn_time = toc

tic
knn_labels_pca = knn(1, train_pca, train_labels, test_pca);
knn_pca_time = toc

tic
knn_labels_mda = knn(1, train_mda, train_labels, test_mda);
knn_mda_time = toc

% Error Rates
knn_error = knn_labels ~= test_labels;
knn_error = sum(knn_error)/test_size

knn_pca_error = knn_labels_pca ~= test_labels;
knn_pca_error = sum(knn_pca_error)/test_size

knn_mda_error = knn_labels_mda ~= test_labels;
knn_mda_error = sum(knn_mda_error)/test_size

