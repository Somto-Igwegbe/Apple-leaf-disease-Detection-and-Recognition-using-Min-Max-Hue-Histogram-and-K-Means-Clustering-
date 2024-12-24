In this study, an efficient image-processing pipeline was explored for detecting 
apple leaf disease. This is implemented in MATLAB. The images undergo preprocessing steps like grayscale conversion, gaussian 
filtering to reduce noise, and contrast stretching to improve image quality. Additionally, 
analysis of the histogram of hue values from several images aided in determining the optimal 
thresholds for spot detection. This was crucial for isolating the regions affected by the disease. 
Feature extraction techniques are applied to detect and capture relevant information in the 
images, followed by k-means clustering to group similar features. The popular elbow method 
is used to ascertain the optimal number of clusters for image segmentation. Subsequently, k
means clustering is applied to segment the images based on their intensity values to identify 
distinct regions. The hue-based spot detection is utilised to isolate areas of interest during 
segmentation.  
