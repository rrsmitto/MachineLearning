function [train_im, train_labels, test_im, test_labels] = load_mnist(train_size, test_size)

    train_im = loadMNISTImages('../data/mnist/train-images-idx3-ubyte');
    train_labels = loadMNISTLabels('../data/mnist/train-labels-idx1-ubyte')';
    
    test_im = loadMNISTImages('../data/mnist/t10k-images-idx3-ubyte');
    test_labels = loadMNISTLabels('../data/mnist/t10k-labels-idx1-ubyte')';
    
    % Shift labels by 1 to allow indexing
    train_labels = train_labels + 1;
    test_labels = test_labels + 1;
    
    train_im = train_im(:,1:train_size);
    train_labels = train_labels(1:train_size);
    
    test_im = test_im(:,1:test_size);
    test_labels = test_labels(1:test_size);
