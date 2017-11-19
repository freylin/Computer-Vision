function features_pos = get_interesting_positive_features(train_path_pos, feature_params)


image_files = dir( fullfile( train_path_pos, '*.jpg') ); %Caltech Faces stored as .jpg
num_images = length(image_files);

D = (feature_params.template_size / feature_params.hog_cell_size)^2 * 31;
features_pos = zeros(num_images * 6, D);

for i = 1 : num_images
    f = image_files(i);
    IM = single(imread(fullfile(f.folder, f.name)));
    HOG = vl_hog(IM, feature_params.hog_cell_size);
    features_pos(i, :) = reshape(HOG, 1, D); 
    
    % mirroring
    IM2 = fliplr(IM);
    HOG = vl_hog(IM2, feature_params.hog_cell_size);
    features_pos(i+num_images, :) = reshape(HOG, 1, D); 
    
    % 3/4 face (right side)
    quarter_width = round(size(IM2, 2) / 4);
    IM2 = IM;
    IM2(:, 1:quarter_width) = 0;
    HOG = vl_hog(IM2, feature_params.hog_cell_size);
    features_pos(i+num_images*2, :) = reshape(HOG, 1, D); 
        
    % 3/4 face (left side)
    IM2 = IM;
    IM2(:, (quarter_width*3+1):end) = 0;
    HOG = vl_hog(IM2, feature_params.hog_cell_size);
    features_pos(i+num_images*3, :) = reshape(HOG, 1, D); 
    
    % 3/4 face (bottom side)
    quarter_height = round(size(IM2, 1) / 4);
    IM2 = IM;
    IM2(1:quarter_height, :) = 0;
    HOG = vl_hog(IM2, feature_params.hog_cell_size);
    features_pos(i+num_images*4, :) = reshape(HOG, 1, D); 
        
    % 3/4 face (top side)
    IM2 = IM;
    IM2((quarter_height*3+1):end, :) = 0;
    HOG = vl_hog(IM2, feature_params.hog_cell_size);
    features_pos(i+num_images*5, :) = reshape(HOG, 1, D); 
    
    %     % half face (right side)
%     half_width = round(size(IM2, 2) / 2);
%     IM2 = IM;
%     IM2(:, 1:half_width) = 0;
%     HOG = vl_hog(IM2, feature_params.hog_cell_size);
%     features_pos(i+num_images*2, :) = reshape(HOG, 1, D); 

%     % shift right face to left
%     IM2(:, 1:half_width) = IM2(:, (half_width+1):end);
%     IM2(:, (half_width+1):end) = 0;    
%     HOG = vl_hog(IM2, feature_params.hog_cell_size);
%     features_pos(i+num_images*3, :) = reshape(HOG, 1, D); 
    
%     % half face (left side)
%     IM2 = IM;
%     IM2(:, (half_width+1):end) = 0;
%     HOG = vl_hog(IM2, feature_params.hog_cell_size);
%     features_pos(i+num_images*4, :) = reshape(HOG, 1, D); 
    
    
%     % shift left face to right
%     IM2(:, (half_width+1):end) = IM2(:, 1:half_width);  
%     IM2(:, 1:half_width) = 0;
%     HOG = vl_hog(IM2, feature_params.hog_cell_size);
%     features_pos(i+num_images*5, :) = reshape(HOG, 1, D); 
end

