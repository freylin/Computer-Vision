% Fundamental Matrix Stencil Code
% CS 4476 / 6476: Computer Vision, Georgia Tech
% Written by Henry Hu

% Returns the camera center matrix for a given projection matrix

% 'Points_a' is nx2 matrix of 2D coordinate of points on Image A
% 'Points_b' is nx2 matrix of 2D coordinate of points on Image B
% 'F_matrix' is 3x3 fundamental matrix

% Try to implement this function as efficiently as possible. It will be
% called repeatly for part III of the project

function [ F_matrix ] = estimate_fundamental_matrix(Points_a,Points_b)

%%%%%%%%%%%%%%%%
% Your code here
% normalization
row = size(Points_a, 1);
mean_a = mean(Points_a);
mean_b = mean(Points_b);
s_a = std2(Points_a - mean_a);
s_b = std2(Points_b - mean_b);
s_mat = [s_a, 0, 0; 0, s_a, 0; 0, 0, 1];
m_mat = [1, 0, -mean_a(1); 0, 1, -mean_a(2); 0, 0, 1];
T_a = s_mat * m_mat;
s_mat(1, 1) = s_b; 
s_mat(2, 2) = s_b;
m_mat(1, 3) = -mean_b(1);
m_mat(2, 3) = -mean_b(2);
T_b = s_mat * m_mat;
point_mat = [Points_a, ones(row, 1)]';
Points_a = (T_a * point_mat)';
point_mat = [Points_b, ones(row, 1)]';
Points_b = (T_b * point_mat)';

% fundamental matrix
A = zeros(row, 8);
for i = 1 : row
    A(i, :) = [Points_a(i,1)*Points_b(i,1), Points_a(i,2)*Points_b(i,1), ...
        Points_b(i,1), Points_a(i,1)*Points_b(i,2), ...
        Points_a(i,2)*Points_b(i,2), Points_b(i,2), ...
        Points_a(i,1), Points_a(i,2)]; 
end
B = -ones(row, 1);
F_matrix = [A \ B; 1];
F_matrix = reshape(F_matrix, [3,3])';

[U, S, V] = svd(F_matrix);
S(end, end) = 0;
F_matrix = U * S * V';
F_matrix = T_b' * F_matrix * T_a;
% disp('The fundamental matrix is:')
% disp(F_matrix);
%%%%%%%%%%%%%%%%

%This is an intentionally incorrect Fundamental matrix placeholder
% F_matrix = [0  0     -.0004; ...
%             0  0      .0032; ...
%             0 -0.0044 .1034];
        
end

