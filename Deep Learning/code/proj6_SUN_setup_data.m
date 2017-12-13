function imdb = proj6_SUN_setup_data()

SceneJPGsPath = '../data/SUN397/';

num_train_per_category = 50;
num_test_per_category  = 50;
categorie_num = 397;
total_images = categorie_num*num_train_per_category + categorie_num * num_test_per_category;

image_size = [64 64]; %downsampling data for speed and because it hurts
% accuracy surprisingly little

imdb.images.data   = zeros(image_size(1), image_size(2), 1, total_images, 'single');
imdb.images.labels = zeros(1, total_images, 'single');
imdb.images.set    = zeros(1, total_images, 'uint8');
image_counter = 1;

categories = {};
fidin=fopen('../data/SUN397/ClassName.txt','r');
nline = 0;
while ~feof(fidin)
    tline = fgetl(fidin);
    nline = nline + 1;
    categories{nline} = tline(2:end);
    if (nline==categorie_num)
        break;
    end
end
fclose(fidin); 
sets = {'train.txt', 'test.txt'};

fprintf('Loading %d train and %d test images from each category\n', ...
          num_train_per_category, num_test_per_category)
fprintf('Each image will be resized to %d by %d\n', image_size(1),image_size(2));

mean_img = zeros(64, 64);
count_train = 0;
%Read each image and resize it to image_size
for set = 1:length(sets)
    img_path = {};
    disp(fullfile('../data/SUN397/',sets{set}));
    fidin=fopen(fullfile('../data/SUN397/',sets{set}),'r');
    nline = 0;
    while ~feof(fidin)
        tline = fgetl(fidin);
        nline = nline + 1;
        img_path{nline} = tline;
    end
    fclose(fidin);
    for category = 1:length(categories)
        cur_images = fullfile('../data/SUN397/',img_path((category-1)*50+1:category*50));
        for i = 1:length(cur_images)
            disp(category);
            disp(i);
            disp(cur_images(i));
            cur_image = imread(fullfile(char(cur_images(i))));
            cur_image = single(cur_image);
            if(size(cur_image,3) > 1)
                cur_image = rgb2gray(cur_image);
            end
            cur_image = imresize(cur_image, image_size);
            if (set == 1)
                count_train = count_train + 1;
                mean_img = mean_img + cur_image;
            end
            % Stack images into a large image_size x 1 x total_images matrix
            % images.data
            imdb.images.data(:,:,1,image_counter) = cur_image;            
            imdb.images.labels(  1,image_counter) = category;
            imdb.images.set(     1,image_counter) = set; %1 for train, 2 for test (val)
            
            image_counter = image_counter + 1;
        end
    end
end

% compute the mean image and then subtract the mean
mean_img = mean2(mean_img/count_train);
imdb.images.data = imdb.images.data - mean_img;

