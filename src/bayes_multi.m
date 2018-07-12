function Classes = myBayes(train_set, train_labels, test_set)
        
    % Class info
    c = max(train_labels);
    [d, n] = size(train_set);

    % Count class frequencies
    class_freq = zeros(1,c);
    for i = 1:n
        class_freq(train_labels(i)) = class_freq(train_labels(i)) + 1;
    end

    % Calculate the mean of train_set for each class
    M = classMean(train_set, train_labels); 

    % Calculate covariance matrices
    C = zeros(d, d, c);
    for i = 1:n
        normalized_sample = train_set(:,i) - M(:,train_labels(i));
        C(:,:,train_labels(i)) = C(:,:,train_labels(i)) + normalized_sample * normalized_sample';
    end
%   
%    for i = 1:c
%        C(:,:,i) = 1 / class_freq(i) * C(:,:,i);
%    end
  
    % Calculate classifier coefficients
    W = zeros(size(C));
    w = zeros(d, c);
    w0 = size(c);

    for i = 1:c
        sigma_inv = pinv(C(:,:,i));
        [~, S, ~] = svd(C(:,:,i));
        sigma_det = sum(sum(S));

        mu = M(:,i);

        W(:,:,i) = -1/2 * sigma_inv;
        w(:,i) = sigma_inv * mu;
        w0(i) = -1/2 * mu' * sigma_inv * mu - 1/2 * log(sigma_det) + log(class_freq(i)/n);
    end

    % Classify the test set 
    discriminators = zeros(1, size(test_set, 2));
    for i = 1:size(test_set, 2)
        maximum = -inf;
        x = test_set(:, i);
        for j = 1:c
            g = x' * W(:,:,j) * x + w(:,j)' * x + w0(j);
            if(g > maximum)
                maximum = g;
                discriminators(i) = j;
            end
        end
    end
    
    Classes = discriminators;

end
