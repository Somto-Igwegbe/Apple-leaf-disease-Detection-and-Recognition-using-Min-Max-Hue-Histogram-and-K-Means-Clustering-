% Read the dataset
somi_dataset = '/MATLAB Drive/Apple rust/rust';
somii = imageDatastore(somi_dataset);


features1 = []
% Visualise two original images and their filtered versions
figure;
for i = 1:2
    % Read original image
    original_img = imread(somii.Files{i});
    
    % Convert image to grayscale and perform contrast stretching
    gray_somi = rgb2gray(original_img);
    stretched_img = imadjust(gray_somi);
    
    % Apply Gaussian filter for noise reduction
    filtered_img = imgaussfilt(stretched_img, 2);
    
    % Plot original image
    subplot(2, 3, 3*(i-1)+1);
    imshow(original_img);
    title(['Original Image ', num2str(i)]);
    
    % Plot filtered image
    subplot(2, 3, 3*(i-1)+2);
    imshow(filtered_img);
    title(['Filtered Image ', num2str(i)]);

    % Plot histogram of hue values
    subplot(2, 3, 3*(i-1)+3);
    hsv_img = rgb2hsv(original_img);
    hue_channel = hsv_img(:,:,1);
    histogram(hue_channel(:), 50);
    xlabel('Hue Value');
    ylabel('Frequency');
    title(['Histogram of Hue Values - Image ', num2str(i)]);
    
    % Extract features using SURF
    gray_img = rgb2gray(original_img);
    points = detectSURFFeatures(gray_img);
    [img_features, ~] = extractFeatures(gray_img, points);
    
    % Append features to the features matrix
    features = [features1; img_features];
end

% Specify the number of clusters
num_clusters = 2;

% Perform k-means clustering
[idx, centroids] = kmeans(features, num_clusters);

% Perform k-means clustering for different numbers of clusters
max_clusters = 10; % Maximum number of clusters to consider
wcss = zeros(max_clusters, 1); % Initialize an array to store within-cluster sum of squares

for k = 1:max_clusters
    [~, centroids, sumd] = kmeans(features, k); % Run k-means clustering
    wcss(k) = sum(sumd); % Compute within-cluster sum of squares and store it
end

% Plot the within-cluster sum of squares (WCSS) as a function of the number of clusters
figure;
plot(1:max_clusters, wcss, 'bo-');
xlabel('Number of Clusters');
ylabel('Within-Cluster Sum of Squares (WCSS)');
sgtitle('Elbow Method for Optimal Number of Clusters');
grid on;

% Define batch size
batch_size = 10; 

% Initialise segmented images cell array
segmented_images = cell(numel(somii.Files), 1);

% Process the images in batches
num_batches = ceil(numel(somii.Files) / batch_size);
for batch = 1:num_batches
    % Determine indices for the current batch
    batch_start = (batch - 1) * batch_size + 1;
    batch_end = min(batch * batch_size, numel(somii.Files));
    batch_indices = batch_start:batch_end;

    % Initialise a cell array to store segmented images for the current batch
    batch_segmented_images = cell(numel(batch_indices), 1);

    % Process images in the current batch
    for i = 1:numel(batch_indices)
        img = imread(somii.Files{batch_indices(i)});
        
        % Extract hue channe
        hsv_img = rgb2hsv(img);
        hue_channel = hsv_img(:,:,1);
        
        % Hue based spot detection
        % Adjust the hue thresholds as needed
        hue_min = 0.05; % Minimum hue value for spots
        hue_max = 0.2;  % Maximum hue value for spots
        spot_mask = (hue_channel >= hue_min) & (hue_channel <= hue_max);
        
        % Convert the spot mask to binary image
        spot_binary = uint8(spot_mask) * 255;
        
        % Performing k-means clustering on the grayscale images
        gray_img = rgb2gray(img);
        intensity_values = double(gray_img(:)); % Ensure intensity_values is double
        [~, centroids] = kmeans(intensity_values, 2); % You can adjust the number of clusters
        
        % Compute distances for all centroids
        distances_all = pdist2(intensity_values, centroids);
        
        % Find the index of the closest centroid for each pixel
        [~, closest_centroid_idx] = min(distances_all, [], 2); % Find minimum distance along rows
        
        % Reshape the segmented image
        segmented_img = reshape(closest_centroid_idx, size(gray_img));
        
        % Apply spot mask to segmented image
        segmented_img(~spot_mask) = 0; % Set non-spot pixels to 0
        
        % Store segmented image in batch_segmented_images cell array
        batch_segmented_images{i} = segmented_img;
    end

    % Store segmented images for the current batch in segmented_images cell array
    segmented_images(batch_indices) = batch_segmented_images;
end

% Initialize cell array to store mappings
image_mapping = cell(numel(somii.Files), 2);

% Visualise segmented images
for i = 1:numel(segmented_images)
    % Get the filename of the original image
    original_filename = somii.Files{i};
    
    % Generate filename for the segmented image
    segmented_filename = ['Segmented_Image_', num2str(i), '.png']; % Adjust the format as needed
    
    % Store the mapping
    image_mapping{i, 1} = original_filename;
    image_mapping{i, 2} = segmented_filename;
    
    fig = figure('Visible', 'off'); % Create figure without displaying it
    imshow(segmented_images{i}, 'DisplayRange', []); % Specify empty display range to auto-adjust
    title(['Segmented Image ', num2str(i), ' - ', original_filename]);
    % Save the figure to a file
    saveas(fig, segmented_filename); % Save segmented image with the generated filename
    close(fig); % Close the figure to release memory
end

% Display a message indicating that the mappings have been created
disp('Mapping between original and segmented images created.');



% Visualiing the original and segmented images side by side
image_path_1 = '/MATLAB Drive/Apple rust/rust/Train_269.jpg';
image_path_2 = '/MATLAB Drive/Apple rust/rust/Train_285.jpg';

% Read the two images
image_1 = imread(image_path_1);
image_2 = imread(image_path_2);

figure;
subplot(2, 2, 1);
imshow(image_1);
title('Original Image 1');

subplot(2, 2, 2);
imshow(image_2);
title('Original Image 2');

% Specify paths to the segmented versions of the images
segmented_path_1 = '/MATLAB Drive/Segmented_Image_67.png';
segmented_path_2 = '/MATLAB Drive/Segmented_Image_73.png';

% Read the segmented images
segmented_image_1 = imread(segmented_path_1);
segmented_image_2 = imread(segmented_path_2);

% Visualize segmented images side by side
subplot(2, 2, 3);
imshow(segmented_image_1);
title('Segmented Image 1');

subplot(2, 2, 4);
imshow(segmented_image_2);
title('Segmented Image 2');

sgtitle('Comparison of Healthy Original and Segmented Images');