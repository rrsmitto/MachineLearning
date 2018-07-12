function M = class_mean(data, labels)

    c = max(labels);
    d = size(data, 1);
    class_freq = zeros(1, c);

    M = zeros(d, c);

    % First sum the data of each class
    for i = 1:size(data, 2)
        M(:, labels(i)) = M(:, labels(i)) + data(:,i);
        class_freq(labels(i)) = class_freq(labels(i)) + 1;
    end

    % Divide the sums by the number of data in each class
    for i = 1:c
        M(:, i) = M(:, i) ./ class_freq(i);
    end
