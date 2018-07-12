%% Support Vector Machine
% In this project I used MatLab's toolbox to classify the MNIST data set of
% hand drawn digits. I used a linear machine and a polynomial machine. Through
% trial and error I found the linear machine to be a decent predictor but the
% polynomial machine was both much faster and more accurate. 

clear all;

%% Load and Process Data 
train_im = loadMNISTImages('../data/mnist/train-images-idx3-ubyte')';
train_labels = loadMNISTLabels('../data/mnist/train-labels-idx1-ubyte')';

test_im = loadMNISTImages('../data/mnist/t10k-images-idx3-ubyte')';
test_labels = loadMNISTLabels('../data/mnist/t10k-labels-idx1-ubyte')';

% Options for training/testing

% Number of training observations, changing this drastically
% changes the run time. 
OBS_TRAIN = 20000;

% Number of test observations, does not significantly affect
% run time
OBS_TEST = 10000;

% Dimension that PCA will reduce to
PCA_DIM = 100;

train_im = train_im(1:OBS_TRAIN, :);
train_labels = train_labels(1:OBS_TRAIN);

test_im = test_im(1:OBS_TEST, :);
test_labels = test_labels(1:OBS_TEST);

coeffs = pca(train_im);

train_im_pca = train_im * coeffs(:,1:PCA_DIM);
test_im_pca = test_im * coeffs(:,1:PCA_DIM);

%% SVM Training and Testing
% This procedure uses PCA reduced data that is fed into the fitcsvm function
% from the MatLab toolbox. A simple description of the overall design is
% there are 10 binary models trained for each class, so that the ith model
% predicts the odds of an observation being in the ith class. The output of
% these 10 binary models is then compared, the model with the maximum score
% determines what class the observation belongs to.

classes = unique(train_labels);

train_labels_bin = [];
test_labels_bin = [];
SVMModels = cell(numel(classes),1);

% Create binary labels for training and testing
for i = 1:numel(classes)
    train_labels_bin = [train_labels_bin; (train_labels == (i-1))];
    test_labels_bin = [test_labels_bin; (test_labels == (i-1))];
end

tic
%% Linear SVM
% Train binary models. This is a linear machine by default. The linear
% machine is decently accurate, it can get under 10% error with only
% 10000 training samples, but takes about 3 minutes to run.
for i = 1:numel(classes)
    SVMModels{i} = fitcsvm(train_im_pca, train_labels_bin(i,:));
end

linear_train_time = toc
tic

% Record scores for each test observation 
scores = zeros(OBS_TEST, numel(classes));
for i = 1:numel(classes)
    [~, score] = predict(SVMModels{i}, test_im_pca);
    scores(:,i) = score(:,2); % Second column contains positive-class scores
end

linear_score_time = toc

% Observations are assigned to the class with the highest score
[~, output_labels] = max(scores, [], 2);
output_labels = output_labels - 1;

% Output error
err_vec = output_labels ~= test_labels';
err = sum(err_vec);
err_percent = err/OBS_TEST*100;

% The linear machine gives decent accuracy but can have very long run times
% with a lot of training observations.
disp(sprintf('Number of training observations: %d', OBS_TRAIN))
disp(sprintf('Number of test observations: %d', OBS_TEST))
disp(sprintf('Error for linear SVM : %f', err_percent))
%}

%% Polynomial SVM
% Repeat training and testing process for a non-linear kernel SVM. I found through trial and
% error that a polynomial of order 2 gives both accurate and very fast results. Compared to the
% linear machine, running 10000 training observations takes 18 seconds with an error of only 3%.
% That's 12% of the time it takes to run the linear machine with the same amount of training observations
% and 7% better error rate.

% Since the polynomial machine runs much faster than the linear machine, all
% training observations can be used.
OBS_TRAIN = 20000;

poly_order = 2;
for i = 1:numel(classes)
    SVMModels{i} = fitcsvm(train_im_pca, train_labels_bin(i,:), 'KernelFunction','polynomial', 'PolynomialOrder', poly_order);
end

poly_train_time = toc
tic

% Record scores for each test observation 
scores = zeros(OBS_TEST, numel(classes));
for i = 1:numel(classes)
    [~, score] = predict(SVMModels{i}, test_im_pca);
    scores(:,i) = score(:,2); % Second column contains positive-class scores
end

poly_score_time = toc

% Observations are assigned to the class with the highest score
[~, output_labels] = max(scores, [], 2);
output_labels = output_labels - 1;

% Output error
err_vec = output_labels ~= test_labels';
err = sum(err_vec);
err_percent = err/OBS_TEST*100;

% The output error for a degree 2 polynomial gives the best performance and runs very fast.
disp(sprintf('Number of training observations: %d', OBS_TRAIN))
disp(sprintf('Number of test observations: %d', OBS_TEST))
disp(sprintf('Error for polynomial SVM of order %i : %f', poly_order, err_percent))
