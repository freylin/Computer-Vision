function build_gmm( image_paths, vocab_size )
% The inputs are 'image_paths', a N x 1 cell array of image paths, and
% 'vocab_size' the size of the vocabulary.
N = size(image_paths, 1);
Features = [];
for i = 1 : N
   img = single(imread(image_paths{i}));
   [~, SIFT_features] = vl_dsift(img, 'step', 30, 'fast');
   Features = [Features, SIFT_features];
end

Features = single(Features);
[means, covariances, priors] = vl_gmm(Features, vocab_size);
save('means.mat', 'means')
save('covariances.mat', 'covariances')
save('priors.mat', 'priors')

end




