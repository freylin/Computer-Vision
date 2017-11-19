% Starter code prepared by James Hays and Sam Birch for Computer Vision

% All of your code will be in "Step 1" and "Step 2", although you can
% modify other parameters in the starter code.

%% Step 0: Set up parameters, vlfeat, category list, and image paths.

%For this project, you will need to report performance for three
%combinations of features / classifiers. It is suggested you code them in
%this order, as well:
% 1) Tiny image features and nearest neighbor classifier
% 2) Bag of sift features and nearest neighbor classifier
% 3) Bag of sift features and linear SVM classifier
%The starter code is initialized to 'placeholder' just so that the starter
%code does not crash when run unmodified and you can get a preview of how
%results are presented.

FEATURE = 'tiny image';
% FEATURE = 'bag of sift';
% FEATURE = 'placeholder';

CLASSIFIER = 'nearest neighbor';
% CLASSIFIER = 'support vector machine';
% CLASSIFIER = 'placeholder';

% set up paths to VLFeat functions. 
% See http://www.vlfeat.org/matlab/matlab.html for VLFeat Matlab documentation
% This should work on 32 and 64 bit versions of Windows, MacOS, and Linux
% run('vlfeat/toolbox/vl_setup')

data_path = '../data/SUN397/'; 

%This is the list of categories / directories to use. The categories are
%somewhat sorted by similarity so that the confusion matrix looks more
%structured (indoor and then urban and then rural).

categories = {};
fidin=fopen('../data/SUN397/ClassName.txt','r');
% fidin=fopen('../data/SUN397/ClassName2.txt','r');
nline = 0;
while ~feof(fidin)
    tline = fgetl(fidin);
    nline = nline + 1;
    categories{nline} = tline(2:end);
end
fclose(fidin);
categories = categories(1:15);
%This list of shortened category names is used later for visualization.
abbr_categories = categories;
    
%number of training examples per category to use. Max is 100. For
%simplicity, we assume this is the number of test cases per category, as
%well.
num_train_per_cat = 50; 
% num_train_per_cat = 100; 
num_categories = length(categories); %number of scene categories.

%This paths for each training and test image.
train_image_paths = cell(num_categories * num_train_per_cat, 1);
test_image_paths  = cell(num_categories * num_train_per_cat, 1);

%The name of the category for each training and test image. With the
%default setup, these arrays will actually be the same, but they are built
%independently for clarity and ease of modification.
train_labels = cell(num_categories * num_train_per_cat, 1);
test_labels  = cell(num_categories * num_train_per_cat, 1);

for i=1:num_categories
   images = dir( fullfile(data_path, categories{i}, '*.jpg'));
   for j=1:num_train_per_cat
       train_image_paths{(i-1)*num_train_per_cat + j} = fullfile(data_path, categories{i}, images(j).name);
       train_labels{(i-1)*num_train_per_cat + j} = categories{i};
   end
   
   images = dir( fullfile(data_path,  categories{i}, '*.jpg'));
   for j=1:num_train_per_cat
       test_image_paths{(i-1)*num_train_per_cat + j} = fullfile(data_path, categories{i}, images(num_train_per_cat + j).name);
       test_labels{(i-1)*num_train_per_cat + j} = categories{i};
   end
end

%% Step 1: Represent each image with the appropriate feature
% Each function to construct features should return an N x d matrix, where
% N is the number of paths passed to the function and d is the 
% dimensionality of each image representation. See the starter code for
% each function for more details.

fprintf('Using %s representation for images\n', FEATURE)

switch lower(FEATURE)    
    case 'tiny image'
        % YOU CODE get_tiny_images.m 
        train_image_feats = get_tiny_images(train_image_paths);
        test_image_feats  = get_tiny_images(test_image_paths);
        
    case 'bag of sift'
        % YOU CODE build_vocabulary.m
        if ~exist('vocab.mat', 'file')
            fprintf('No existing visual word vocabulary found. Computing one from training images\n')
            vocab_size = 1000; %Larger values will work better (to a point) but be slower to compute
            vocab = build_vocabulary(train_image_paths, vocab_size);
            save('vocab.mat', 'vocab')
        end
        
        % YOU CODE get_bags_of_sifts.m
%         train_image_feats = get_bags_of_sifts(train_image_paths);
%         test_image_feats  = get_bags_of_sifts(test_image_paths);
        if ~exist('train_image_feats.mat', 'file')
            train_image_feats = get_bags_of_sifts(train_image_paths);
            test_image_feats  = get_bags_of_sifts(test_image_paths);
            save('train_image_feats.mat', 'train_image_feats');
            save('test_image_feats.mat', 'test_image_feats');
        end
        load('train_image_feats.mat');
        load('test_image_feats.mat');
        
    case 'placeholder'
        train_image_feats = [];
        test_image_feats = [];
        
    otherwise
        error('Unknown feature type')
end

% If you want to avoid recomputing the features while debugging the
% classifiers, you can either 'save' and 'load' the features as is done
% with vocab.mat, or you can utilize Matlab's "code sections" functionality
% http://www.mathworks.com/help/matlab/matlab_prog/run-sections-of-programs.html

%% Step 2: Classify each test image by training and using the appropriate classifier
% Each function to classify test features will return an N x 1 cell array,
% where N is the number of test cases and each entry is a string indicating
% the predicted category for each test image. Each entry in
% 'predicted_categories' must be one of the 15 strings in 'categories',
% 'train_labels', and 'test_labels'. See the starter code for each function
% for more details.

fprintf('Using %s classifier to predict test set categories\n', CLASSIFIER)

switch lower(CLASSIFIER)    
    case 'nearest neighbor'
        % YOU CODE nearest_neighbor_classify.m 
        predicted_categories = nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats);
        
    case 'support vector machine'
        % YOU CODE svm_classify.m 
        predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats);
        
    case 'placeholder'
        %The placeholder classifier simply predicts a random category for
        %every test case
        random_permutation = randperm(length(test_labels));
        predicted_categories = test_labels(random_permutation); 
        
    otherwise
        error('Unknown classifier type')
end



%% Step 3: Build a confusion matrix and score the recognition system
% You do not need to code anything in this section. 

% If we wanted to evaluate our recognition method properly we would train
% and test on many random splits of the data. You are not required to do so
% for this project.

% This function will recreate results_webpage/index.html and various image
% thumbnails each time it is called. View the webpage to help interpret
% your classifier performance. Where is it making mistakes? Are the
% confusions reasonable?
create_results_webpage( train_image_paths, ...
                        test_image_paths, ...
                        train_labels, ...
                        test_labels, ...
                        categories, ...
                        abbr_categories, ...
                        predicted_categories)