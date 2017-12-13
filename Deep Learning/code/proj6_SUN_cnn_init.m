function net = proj6_SUN_cnn_init()
%code for Computer Vision, Georgia Tech by James Hays
%based of the MNIST example from MatConvNet

rng('default');
rng(0);

% constant scalar for the random initial network weights. You shouldn't
% need to modify this.
f=1/100; 

net.layers = {} ;

net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(9,9,1,10, 'single'), zeros(1, 10, 'single')}}, ...
                           'stride', 1, ...
                           'pad', 0, ...
                           'name', 'conv1') ;

net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [7 7], ...
                           'stride', 3, ...
                           'pad', 0) ;

net.layers{end+1} = struct('type', 'relu') ;
       
% insert a convolutional layer after the existing relu layer 
% with a 5x5 spatial support 
% followed by a max-pool over a 3x3 window with a stride of 2
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(5,5,1,20, 'single'), zeros(1, 20, 'single')}}, ...
                           'stride', 1, ...
                           'pad', 0, ...
                           'name', 'conv1') ;

net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 0) ;
                      
net.layers{end+1} = struct('type', 'relu') ;

% dropout regularization
net.layers{end+1} = struct('type', 'dropout', ...
                           'rate', 0.5);
                       
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(6,6,20,397, 'single'), zeros(1, 397, 'single')}}, ...
                           'stride', 1, ...
                           'pad', 0, ...
                           'name', 'fc1') ;
                   
                      
% Loss layer
net.layers{end+1} = struct('type', 'softmaxloss') ;

net = vl_simplenn_tidy(net);

% %You can insert batch normalization layers here
net = insertBnorm(net, 1);
net = insertBnorm(net, 5);

% Visualize the network
vl_simplenn_display(net, 'inputSize', [64 64 1 50])


% --------------------------------------------------------------------
function net = insertBnorm(net, layer_index)
% --------------------------------------------------------------------
assert(isfield(net.layers{layer_index}, 'weights'));
ndim = size(net.layers{layer_index}.weights{1}, 4);
layer = struct('type', 'bnorm', ...
               'weights', {{ones(ndim, 1, 'single'), zeros(ndim, 1, 'single')}}, ...
               'learningRate', [1 1 0.05]) ;
net.layers{layer_index}.weights{2} = [] ;  % eliminate bias in previous conv layer
net.layers = horzcat(net.layers(1:layer_index), layer, net.layers(layer_index+1:end)) ;



