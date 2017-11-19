function K = chiSquared(mat1, mat2)
    for i = 1 : size(mat2, 1)
        d = mat1 - mat2(i, :);
        s = mat1 + mat2(i, :);
        K(:, i) = sum(d.^2 ./ (s/2+eps), 2);
    end
    K = 1 - K;
 