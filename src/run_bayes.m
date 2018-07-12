%% Load and Vectorize Data

train_size = 20000;
test_size = 10000;
[train_im, train_labels, test_im, test_labels] = load_mnist(train_size, test_size);

%% Dimensionality Reduction

dprime = 300;

% PCA

tic
principle_coeffs = my_pca(train_im);
pca_time = toc

principle_coeffs = principle_coeffs(1:dprime,:);

train_pca = principle_coeffs * train_im;
test_pca = principle_coeffs * test_im;

% MDA
tic
mda_coeffs = mda(train_im, train_labels, 1);
mda_time = toc

mda_coeffs = mda_coeffs(1:dprime,:);

train_mda = mda_coeffs * train_im;
test_mda = mda_coeffs * test_im;

%% Classification
% Baye's Classification

tic
bayes_labels = bayes_multi(train_im, train_labels, test_im);
bayes_time = toc

tic
bayes_labels_pca = bayes_multi(train_pca, train_labels, test_pca);
bayes_pca_time = toc

tic
bayes_labels_mda = bayes_multi(train_mda, train_labels, test_mda);
bayes_mda_time = toc

%% Error Rates

bayes_error = bayes_labels ~= test_labels;
bayes_error = sum(bayes_error) / test_size

bayes_pca_error = bayes_labels_pca ~= test_labels;
bayes_pca_error = sum(bayes_pca_error) / test_size

bayes_mda_error = bayes_labels_mda ~= test_labels;
bayes_mda_error = sum(bayes_mda_error) / test_size


