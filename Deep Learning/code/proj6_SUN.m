function [net, info] = proj6_SUN()
%code for Computer Vision, Georgia Tech by James Hays
%based off the MNIST and CIFAR examples from MatConvNet

run(fullfile('..','matconvnet-1.0-beta25', 'matlab', 'vl_setupnn.m')) ;

%It might actually be problematic to run vl_setup, because VLFeat has a
%version of vl_argparse that conflicts with the matconvnet version. You
%shouldn't need VLFeat for this project.
% run(fullfile('vlfeat-0.9.20', 'toolbox', 'vl_setup.m'));

%opts.expDir is where trained networks and plots are saved.
opts.expDir = fullfile('..','data','part3') ;

%opts.batchSize is the number of training images in each batch. You don't
%need to modify this.
opts.batchSize = 50 ;

% opts.learningRate is a critical parameter that can dramatically affect
% whether training succeeds or fails. For most of the experiments in this
% project the default learning rate is safe.
opts.learningRate = 0.001 ;

% opts.numEpochs is the number of epochs. If you experiment with more
% complex networks you might need to increase this. Likewise if you add
% regularization that slows training.
opts.numEpochs = 100 ;

% % An example of learning rate decay as an alternative to the fixed learning
% % rate used by default. This isn't necessary but can lead to better
% % performance.
% opts.learningRate = logspace(-3, -4, 100) ;
% opts.numEpochs = numel(opts.learningRate) ;

%opts.continue controls whether to resume training from the furthest
%trained network found in opts.batchSize. If you want to modify something
%mid training (e.g. learning rate) this can be useful. You might also want
%to resume a network that hit the maximum number of epochs if you think
%further training can improve accuracy.
opts.continue = false ;

%GPU support is off by default.
% opts.gpus = [] ;

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

% The cnn_init function specifies the network architecture. You will be
% modifying the function.
net = proj6_SUN_cnn_init();

% The setup_data function loads the training and testing images into
% MatConvNet's imdb structure. You will be modifying the function.

% The commented out code can cache the image database so it isn't rebuilt
% with each run. I found it fast enough to rebuild and less likely to cause
% errors when you change the way images are preprocessed.

imdb_filename = 'sun.mat';
if exist(imdb_filename, 'file')
  imdb = load(imdb_filename) ;
else
  imdb = proj6_SUN_setup_data();
  save(imdb_filename, '-struct', 'imdb') ;
end



%% -------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

[net, info] = cnn_train(net, imdb, @getBatch, opts, ...
    'val', find(imdb.images.set == 2)) ;

save('sun_net', '-struct', 'net') ;

fprintf('Lowest validation erorr is %f\n', min(cat(1, info.val(:).top1err)))
end

% --------------------------------------------------------------------
function [im, labels] = getBatch(imdb, batch)

im = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;

img_num = size(batch, 2);
rand_index = randi(img_num, 1, img_num/2);
im_flip = im;
im_flip(:, :, :, rand_index) = fliplr(im(:, :, :, rand_index));
im = im_flip;
end