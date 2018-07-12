function Vs = mda(data, labels, energy_ratio)

    c = max(labels);
    [d, n] = size(data);
    class_freq = zeros(1, c);

    % Total mean of data vectors
    m = mean(data, 2); 
    % Mean of each class
    M = classMean(data, labels); 
    
    %% In-class scatter
    means = [];
    for i = 1:size(labels, 2)
        means = [means, M(labels(i))];
    end
    % Normalize
    normalized_data = data - means; 
    % Compute scatter
    Sw = zeros(size(data, 1));
    for i = 1:n
        Sw = normalized_data * normalized_data';
    end

    %% Between-class scatter    
    Sb = zeros(d, d);
    for i = 1:n
        class_freq(labels(i)) = class_freq(labels(i)) + 1;
    end
    % Sb is the sum of the outer products m_i - m, m_i = class mean, m = total mean
    for i = 1:c
        Sb = Sb + class_freq(i) * (M(:,i) - m) * (M(:,i) - m)';
    end

    % Solve Sb * wi = Sw * lambda * wi
    [V, D] = eig(Sb, Sw, 'qz');

    [~, ind] = sort(diag(D), 'descend');

    Ds = D(ind,ind);
    Vs = V(:,ind);

    total_energy = 0;
    initial_ind = 0;
    for i = 1:d
        if (~isnan(Ds(i,i))) & (Ds(i,i) ~= Inf)
            if initial_ind == 0
                initial_ind = i;
            end
            total_energy = total_energy + Ds(i,i);
        end
    end

    current_energy = 0;
    count = 1;
    while (current_energy / total_energy) < energy_ratio;
        if (~isnan(Ds(count,count))) & (Ds(count,count) ~= Inf)
            current_energy = current_energy + Ds(count,count);
        end
        count = count + 1;
    end

    mda_coeffs = Vs(initial_ind:count,:);

