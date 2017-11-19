% Sliding window face detection with linear SVM. 
% Base code by James Hays, except for pieces of evaluation code from Pascal
% VOC toolkit. Images from CMU+MIT face database, CalTech Web Face
% Database, and SUN scene database.

close all
clear
run('/Users/linjin/Desktop/CV/P3/vlfeat/toolbox/vl_setup.m')

[~,~,~] = mkdir('visualizations');

data_path = '../data/'; 
train_path_pos = fullfile(data_path, 'cars_brad'); %Positive training examples. 36x36 head crops
non_face_scn_path = fullfile(data_path, 'background'); %We can mine random or hard negatives from here
test_scn_path = fullfile(data_path,'cars_markus');

%The faces are 36x36 pixels, which works fine as a template size. You could
%add other fields to this struct if you want to modify HoG default
%parameters such as the number of orientations, but that does not help
%performance in our limited test.
feature_params = struct('template_size', 36, 'hog_cell_size', 6);


%% Step 1. Load positive training crops and random negative examples
image_files = dir( fullfile( train_path_pos, '*.jpg') ); %Caltech Faces stored as .jpg
num_images = length(image_files);

D = (feature_params.template_size / feature_params.hog_cell_size)^2 * 31;
features_pos = zeros(num_images, D);

for i = 1 : num_images
    f = image_files(i);
    IM = imread(fullfile(f.folder, f.name));
    IM = rgb2gray(IM);
    IM = imresize(IM, [36, 36]);
    IM = single(IM);
    HOG = vl_hog(IM, feature_params.hog_cell_size);
    features_pos(i, :) = reshape(HOG, 1, D); 
end

num_negative_examples = 10000; %Higher will work strictly better, but you should start with 10000 for debugging
features_neg = get_random_negative_features( non_face_scn_path, feature_params, num_negative_examples);

    
%% step 2. Train Classifier

X = [features_pos; features_neg];
Y = [ones(size(features_pos, 1), 1); -ones(size(features_neg, 1), 1)];
lambda = 0.0001;
[w, b] = vl_svmtrain(X', Y', lambda);

%% step 3. Examine learned classifier
% You don't need to modify anything in this section. The section first
% evaluates _training_ error, which isn't ultimately what we care about,
% but it is a good sanity check. Your training error should be very low.

fprintf('Initial classifier performance on train data:\n')
confidences = [features_pos; features_neg]*w + b;
label_vector = [ones(size(features_pos,1),1); -1*ones(size(features_neg,1),1)];
[tp_rate, fp_rate, tn_rate, fn_rate] =  report_accuracy( confidences, label_vector );

% Visualize how well separated the positive and negative examples are at
% training time. Sometimes this can idenfity odd biases in your training
% data, especially if you're trying hard negative mining. This
% visualization won't be very meaningful with the placeholder starter code.
non_face_confs = confidences( label_vector < 0);
face_confs     = confidences( label_vector > 0);
figure(2); 
plot(sort(face_confs), 'g'); hold on
plot(sort(non_face_confs),'r'); 
plot([0 size(non_face_confs,1)], [0 0], 'b');
hold off;

% Visualize the learned detector. This would be a good thing to include in
% your writeup!
n_hog_cells = sqrt(length(w) / 31); %specific to default HoG parameters
imhog = vl_hog('render', single(reshape(w, [n_hog_cells n_hog_cells 31])), 'verbose') ;
figure(3); imagesc(imhog) ; colormap gray; set(3, 'Color', [.988, .988, .988])

pause(0.1) %let's ui rendering catch up
hog_template_image = frame2im(getframe(3));
% getframe() is unreliable. Depending on the rendering settings, it will
% grab foreground windows instead of the figure in question. It could also
% return a partial image.
imwrite(hog_template_image, 'visualizations/hog_template.png')
    
 
%% step 4. (optional extra credit) Mine hard negatives


%% Step 5. Run detector on test set.
% YOU CODE 'run_detector'. Make sure the outputs are properly structured!
% They will be interpreted in Step 6 to evaluate and visualize your
% results. See run_detector.m for more details.
[bboxes, confidences, image_ids] = run_detector(test_scn_path, w, b, feature_params);

% run_detector will have (at least) two parameters which can heavily
% influence performance -- how much to rescale each step of your multiscale
% detector, and the threshold for a detection. If your recall rate is low
% and your detector still has high precision at its highest recall point,
% you can improve your average precision by reducing the threshold for a
% positive detection.


%% Step 6. Evaluate and Visualize detections
% These functions require ground truth annotations, and thus can only be
% run on the CMU+MIT face test set. Use visualize_detectoins_by_image_no_gt
% for testing on extra images (it is commented out below).

% Don't modify anything in 'evaluate_detections'!
% [gt_ids, gt_bboxes, gt_isclaimed, tp, fp, duplicate_detections] = ...
%     evaluate_detections(bboxes, confidences, image_ids, label_path);

% visualize_detections_by_image(bboxes, confidences, image_ids, tp, fp, test_scn_path, label_path)
visualize_detections_by_image_no_gt(bboxes, confidences, image_ids, test_scn_path)

% visualize_detections_by_confidence(bboxes, confidences, image_ids, test_scn_path, label_path);

% performance to aim for
% random (stater code) 0.001 AP
% single scale ~ 0.2 to 0.4 AP
% multiscale, 6 pixel cell size and detector step ~ 0.83 AP
% multiscale, 4 pixel cell size and detector step ~ 0.89 AP
% multiscale, 3 pixel cell size and detector step ~ 0.92 AP