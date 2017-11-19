function image_feats = get_fisher(image_paths)

load('means.mat')
load('covariances.mat')
load('priors.mat')
N = size(image_paths, 1);
image_feats = [];
for i = 1 : N
    if (mod(i, 100) == 0)
        disp(i);
    end
    img = single(imread(image_paths{i}));
    [~, SIFT_features] = vl_dsift(img, 'step', 3,  'fast');
    SIFT_features = single(SIFT_features);
    feature = vl_fisher(SIFT_features, means, covariances, priors);
    image_feats = [image_feats; feature'];
end