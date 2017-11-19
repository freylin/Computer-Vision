function output = my_imfilter(image, filter)
% This function is intended to behave like the built in function imfilter()
% See 'help imfilter' or 'help conv2'. While terms like "filtering" and
% "convolution" might be used interchangeably, and they are indeed nearly
% the same thing, there is a difference:
% from 'help filter2'
%    2-D correlation is related to 2-D convolution by a 180 degree rotation
%    of the filter matrix.

% Your function should work for color images. Simply filter each color
% channel independently.

% Your function should work for filters of any width and height
% combination, as long as the width and height are odd (e.g. 1, 7, 9). This
% restriction makes it unambigious which pixel in the filter is the center
% pixel.

% Boundary handling can be tricky. The filter can't be centered on pixels
% at the image boundary without parts of the filter being out of bounds. If
% you look at 'help conv2' and 'help imfilter' you see that they have
% several options to deal with boundaries. You should simply recreate the
% default behavior of imfilter -- pad the input image with zeros, and
% return a filtered image which matches the input resolution. A better
% approach is to mirror the image content over the boundaries for padding.

% % Uncomment if you want to simply call imfilter so you can see the desired
% % behavior. When you write your actual solution, you can't use imfilter,
% % filter2, conv2, etc. Simply loop over all the pixels and do the actual
% % computation. It might be slow.
% output = imfilter(image, filter);


%%%%%%%%%%%%%%%%
% Your code here
%%%%%%%%%%%%%%%%

filter_height = size(filter, 1);
filter_width = size(filter, 2);

%both dimensions of the filter should be odd
if (mod(filter_height, 2) == 0 || mod(filter_width, 2) ==0 )
    error('Wrong filter! Both dimensions shoule be odd!');
end;

% pad the input image with zeros
pad_rows = (filter_height - 1) / 2;
pad_cols = (filter_width - 1) / 2;
padded_image = padarray(image, [pad_rows, pad_cols]);

image_height = size(image, 1);
image_width = size(image, 2);
% count how many color channels the input has
num_colors = size(image,3); 
% filter each color channel
result_channel(1:image_height, 1:image_width) = 1;
for n = 1:num_colors
    image_channel = padded_image(:, :, n);
    for i = 1:image_height
        ii = i + filter_height - 1;
        for j = 1:image_width
            jj = j + filter_width - 1;
            result_channel(i,j) = sum(sum(image_channel(i:ii, j:jj).* filter));
        end;
    end;
    % merge
    if (n == 1)
        output = result_channel;
    else
        output = cat(3, output, result_channel);
    end;
end;

