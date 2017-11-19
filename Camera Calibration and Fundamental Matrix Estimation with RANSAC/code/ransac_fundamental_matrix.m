% RANSAC Stencil Code
% CS 4476 / 6476: Computer Vision, Georgia Tech
% Written by Henry Hu

% Find the best fundamental matrix using RANSAC on potentially matching
% points

% 'matches_a' and 'matches_b' are the Nx2 coordinates of the possibly
% matching points from pic_a and pic_b. Each row is a correspondence (e.g.
% row 42 of matches_a is a point that corresponds to row 42 of matches_b.

% 'Best_Fmatrix' is the 3x3 fundamental matrix
% 'inliers_a' and 'inliers_b' are the Mx2 corresponding points (some subset
% of 'matches_a' and 'matches_b') that are inliers with respect to
% Best_Fmatrix.

% For this section, use RANSAC to find the best fundamental matrix by
% randomly sample interest points. You would reuse
% estimate_fundamental_matrix() from part 2 of this assignment.

% If you are trying to produce an uncluttered visualization of epipolar
% lines, you may want to return no more than 30 points for either left or
% right images.

function [ Best_Fmatrix, inliers_a, inliers_b] = ransac_fundamental_matrix(matches_a, matches_b)


%%%%%%%%%%%%%%%%
% Your code here
row = size(matches_a, 1);
add_mat = ones(row, 1);
matches_a_added = [matches_a, add_mat];
matches_b_added = [matches_b, add_mat];

threshold = 0.08;
% compute iterate times
sample_num = 8;
probability = 0.999;
outlier_ratio = 0.6;
N = round(log(1-probability) / log(1-(1-outlier_ratio)^sample_num));
disp(N);
% compute Fmatrix wrt each sample
max_num = 0;
Best_Fmatrix = zeros(3, 3);
for i = 1 : N
%    if mod(i, 10000) == 0
%        disp(i);
%    end
   index = randsample(row, sample_num);
   Fmatrix = estimate_fundamental_matrix(matches_a(index, :), matches_b(index, :));
   count = 0;
   for j = 1 : row
       if abs(matches_b_added(j, :) * Fmatrix * matches_a_added(j, :)') <= threshold
           count = count + 1;
       end
   end
   if count > max_num
       max_num = count;
       Best_Fmatrix = Fmatrix;
   end
end

% record the top matches
dist_mat = zeros(row, 1);
for i = 1 : row
   dist_mat(i, 1) = abs(matches_b_added(i, :) * Best_Fmatrix * matches_a_added(i, :)');
end
[~, ind] = sort(dist_mat);
% num = 100;
num = row;
while dist_mat(ind(num,1), 1) > threshold
    num = num - 1;
    if num == 0
        error('No inlier!');
    end
end
disp(num);
inliers_a = matches_a(ind(1:num), :);
inliers_b = matches_b(ind(1:num), :);
%%%%%%%%%%%%%%%%

% Your ransac loop should contain a call to 'estimate_fundamental_matrix()'
% that you wrote for part II.

%placeholders, you can delete all of this
% Best_Fmatrix = estimate_fundamental_matrix(matches_a(1:10,:), matches_b(1:10,:));
% inliers_a = matches_a(1:30,:);
% inliers_b = matches_b(1:30,:);

end

