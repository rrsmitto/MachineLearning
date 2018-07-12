function Classes = knn(k, train_set, train_labels, test_set)
   
    test_size = size(test_set);
    train_size = size(train_set);

    Classes = [];

    for i = 1:test_size(2)
        
        % Find the square difference for the ith test vector
        diff = train_set - diag(test_set(:,i)) * ones(train_size);
        squares = diff .^ 2;
        squares_sum = sum(squares);

        [~, index]  = sort(squares_sum);

        % Classify the target sample to the class with the most nearest neighbors
        class = mode(index(1:k));
        Classes = [Classes, train_labels(class)];
    end

end
