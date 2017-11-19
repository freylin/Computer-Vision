function get_pca_mat(scale_factor, feature_width, dim)
    imgs = dir('../data/pca_images/*.jpg');
    feature_matrix = [];
    for i = 1 : size(imgs, 1)
        image = imread(strcat('../data/pca_images/', imgs(i).name));
        image = single(image) / 255;
        image = imresize(image, scale_factor, 'bilinear');
        image_bw = rgb2gray(image);
        [x, y] = get_interest_points(image_bw, feature_width);
        [image_features] = get_features(image_bw, x, y, feature_width);
        feature_matrix = [feature_matrix; image_features];
    end

    [coeff, ~, ~, ~, ~, mu] = pca(feature_matrix, 'Centered', true, 'NumComponents', dim); 
    % save pca_32 coeff mu
    % save pca_64 coeff mu
    save pca coeff mu
end