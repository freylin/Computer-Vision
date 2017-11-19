% Local Feature Stencil Code
% CS 4476 / 6476: Computer Vision, Georgia Tech
% Written by James Hays

% 'features1' and 'features2' are the n x feature dimensionality features
%   from the two images.
% If you want to include geometric verification in this stage, you can add
% the x and y locations of the interest points as additional features.
%
% 'matches' is a k x 2 matrix, where k is the number of matches. The first
%   column is an index in features1, the second column is an index
%   in features2. 
% 'Confidences' is a k x 1 matrix with a real valued confidence for every
%   match.
% 'matches' and 'confidences' can empty, e.g. 0x2 and 0x1.
function [matches, confidences] = match_features_pca(features1, features2)

% This function does not need to be symmetric (e.g. it can produce
% different numbers of matches depending on the order of the arguments).

% To start with, simply implement the "ratio test", equation 4.18 in
% section 4.1.3 of Szeliski. For extra credit you can implement various
% forms of spatial verification of matches.

% use pca to create a low dimensional descriptor
% load pca_32.mat coeff mu
load pca_64.mat coeff mu
features1 = (features1 - mu) * coeff;
features2 = (features2 - mu) * coeff;

% use kd-tree to count the nearest neighbours
[ind, dist] = knnsearch(features2, features1, 'K', 2, 'NSMethod', 'kdtree');
cfd = (dist(:, 1) ./ dist(:, 2));
ind = ind(:,1);

matches = [];
confidences = [];
for i = 1 : size(features1, 1)
    matches = [matches; i, ind(i)];
    confidences = [confidences; 1-cfd(i)];
end

% Sort the matches so that the most confident onces are at the top of the
% list. You should probably not delete this, so that the evaluation
% functions can be run on the top matches easily.
[confidences, ind] = sort(confidences, 'descend');
if (size(confidences,1)>100)
    confidences = confidences(1:100, :);
    ind = ind(1:100, :);
end
matches = matches(ind, :);