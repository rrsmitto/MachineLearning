function [V, D] = my_pca(X)

    % Mean vector and a matrix of the mean vector repeated for subtraction
    m = mean(X, 2);

    % Normalize X
    X = X - m;
   
    % Scatter matrix
    S = cov(X');

    % Return eignvectors sorted corresponding to the greatest eigenvalues
    [V, D] = eig(S);
    [D, i] = sort(diag(D), 'descend');
    V = V(:,i);

end
