# Airbus-Ship-Detection-Challenge
 I am solving a Kaggle task at https://www.kaggle.com/competitions/airbus-ship-detection, where the goal is to build masks for ship detection. The foundation of the solution is a U-net neural network architecture.
# Load data

To download the dataset, please follow the link https://www.kaggle.com/competitions/airbus-ship-detection/data and extract the archive into the "data" folder. Once done, you can run the "train_and_analysis.ipynb" file.

# Idea
The main challenges of my solution were time and class imbalance. In our dataset, ships occupied a small percentage of the area, and many images didn't contain any ships at all. To expedite the training process, I adopted the following strategy. I divided our images into 9 equal squares and trained the model only on the data (tensors with dimensions of 256x256x3) where the corresponding masks had values. By doing so, we reduced the class imbalance and significantly accelerated the training process.
# Model
You can look at my model using model.summary() . Main components is:
## conv block: 
A convolutional block composed of two convolutional layers with batch normalization and ReLU activation. It takes input inputs, applies convolutions, and returns the output.

## upconv block: 
An upsampling convolutional block that combines the upsampled features with skip connections from the encoder path. It performs transposed convolutions, concatenates the features, applies convolutions, and returns the output.

## attention gate: 
An attention gate that enhances the spatial information flow by calculating attention weights. It takes the features x and a gating signal g, applies convolutions, computes attention weights, and applies element-wise multiplication to the input features.

# Here are some ideas for improvement:

Dynamic thresholding: Instead of using a fixed threshold of 0.7 for considering pixels, calculate the threshold dynamically for each image based on the statistics of the ship area. This can help account for variations in lighting and other factors that may affect the predictions on different images.

DBSCAN for homogeneous backgrounds: Utilize DBSCAN (Density-Based Spatial Clustering of Applications with Noise) to identify homogeneous backgrounds in the images. By clustering pixels with similar characteristics, you can potentially separate the background from the foreground more accurately.

Selecting rectangular regions: Currently, the model predicts ship masks even for islands, ports, and other irregular shapes. Consider selecting only small regions that have a closer approximation to rectangular shape. This can help filter out false positives and focus on regions that are more likely to be ships.
