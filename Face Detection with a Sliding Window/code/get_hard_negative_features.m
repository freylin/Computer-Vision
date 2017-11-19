function features_hard_neg = get_hard_negative_features(non_face_scn_path, w, b, feature_params, num_samples)

test_scenes = dir( fullfile( non_face_scn_path, '*.jpg' ));
t_size = feature_params.template_size;
c_size = feature_params.hog_cell_size;
D = (t_size / c_size)^2 * 31;
b_size = t_size / c_size;
iters = 20;
scale_factor = 0.9;
threshold = 0.0;
features_hard_neg = [];
k = 0;
kmax = 0;
num_samples_each = ceil(num_samples / length(test_scenes));

for i = 1 : length(test_scenes)
    kmax = kmax + num_samples_each;
    fprintf('Getting hard negative features in %s\n', test_scenes(i).name)
    img = imread( fullfile( non_face_scn_path, test_scenes(i).name ));
    img = single(img)/255;
    if(size(img,3) > 1)
        img = rgb2gray(img);
    end
    
    size_ratio = 1;
    for j = 1 : iters
        if (k >= kmax)
            break;
        end
        HOG = vl_hog(img, c_size);
        n = size(HOG, 1);
        m = size(HOG, 2);
        for y = 1 : (n-b_size+1)
            for x = 1 : (m-b_size+1)
                if (k >= kmax)
                    break;
                end
                sample_hog = HOG(y:(y+b_size-1), x:(x+b_size-1), :);
                feature = reshape(sample_hog, 1, D);
                conf = feature * w + b;
                if (conf > threshold)
                    features_hard_neg = [features_hard_neg; feature];
                    k = k + 1;
                end
            end
        end
        img = imresize(img, scale_factor);
        size_ratio = size_ratio * scale_factor;
    end
    disp(k);
end