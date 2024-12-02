# Recognition of Land and Water from Satellite Images using U-Net

#### [Read that blog with GitHub Repository on Global News One ](https://globalnewsone.com/recognition-of-land-and-water-from-satellite-images-using-u-net/)

## Importing Libraries and Modules

The initial step involves importing all necessary libraries and modules required for the project. Here's a detailed explanation of each import and its purpose:

---

### **1. TensorFlow**
```python
import tensorflow as tf
```
- **Purpose**: TensorFlow is the primary deep learning framework used to build and train the U-Net model.
- **Why TensorFlow?**
  - Provides a flexible API for building complex neural networks.
  - Offers built-in support for GPU acceleration, which speeds up training for large datasets like satellite imagery.

---

### **2. Matplotlib**
```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
```
- **Purpose**: 
  - `matplotlib.pyplot`: Used for visualizing the training process, including accuracy/loss graphs, and displaying images and their segmentation masks.
  - `matplotlib.image`: Handles image-related functionalities like loading and displaying images.

---

### **3. OpenCV**
```python
import cv2
```
- **Purpose**: OpenCV is used for efficient image preprocessing tasks like resizing and format conversions.
- **Why OpenCV?**
  - Optimized for high-performance computer vision tasks.
  - Handles large image datasets effectively, which is crucial for satellite imagery processing.

---

### **4. NumPy**
```python
import numpy as np
```
- **Purpose**: NumPy is essential for handling numerical computations and working with multidimensional arrays.
- **Why NumPy?**
  - TensorFlow relies on NumPy arrays for input/output.
  - Useful for preprocessing steps, such as normalization and reshaping.

---

### **5. OS Module**
```python
import os
```
- **Purpose**: The `os` module allows interaction with the file system to:
  - Navigate directories.
  - Access image and mask files stored in the dataset.

---

### **6. skimage (scikit-image)**
```python
from skimage.io import imread, imshow
from skimage.transform import resize
```
- **Purpose**:
  - `imread` and `imshow`: Handle loading and displaying images.
  - `resize`: Used for resizing images to a uniform dimension for model input.
- **Why skimage?**
  - Lightweight and specifically optimized for image processing tasks.

---

### **7. Keras (from TensorFlow)**
```python
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
```
- **Purpose**: Keras layers and utilities are used to build, compile, and train the U-Net model.
- **Breakdown**:
  - **Layers**:
    - `Input`: Defines the input shape of the model.
    - `Conv2D` and `Conv2DTranspose`: Perform convolution and transposed convolution for downsampling and upsampling.
    - `MaxPooling2D`: Reduces spatial dimensions during the encoding process.
    - `UpSampling2D`: Restores spatial dimensions during decoding.
    - `concatenate`: Merges encoder and decoder layers (skip connections).
    - `BatchNormalization`: Normalizes layer outputs to improve training stability.
    - `Dropout`: Reduces overfitting by randomly dropping connections during training.
    - `Lambda`: Allows custom operations during model definition.
  - **Model**:
    - `Model`: Combines input and layers into a functional neural network.
    - `save_model`: Saves the trained model for future use.
  - **Callbacks**:
    - `EarlyStopping`: Stops training when validation performance stops improving.
    - `ModelCheckpoint`: Saves the best-performing model during training.

---

### Why These Libraries?
The combination of these libraries provides a powerful toolkit for handling:
1. **Data Processing**: OpenCV, skimage, and NumPy for loading, resizing, and normalizing satellite images.
2. **Visualization**: Matplotlib for displaying images and tracking model performance.
3. **Deep Learning**: TensorFlow/Keras for building and training the U-Net model with robust optimization and callbacks.

## Downloading the Dataset Using KaggleHub

*   List item
*   List item



---

### **Code**:
```python
# Download the dataset using KaggleHub
import kagglehub
path = kagglehub.dataset_download("franciscoescobar/satellite-images-of-water-bodies")
print("Path to dataset files:", path)
```

---

### **Purpose**:
This step is dedicated to fetching the dataset required for training and evaluating the segmentation model.

1. **Dataset Source**:
   - The dataset, **"Satellite Images of Water Bodies"**, is hosted on Kaggle.
   - It contains satellite images and their corresponding masks that label areas of water.

2. **Why KaggleHub?**:
   - KaggleHub simplifies the process of downloading datasets directly into the script or notebook without needing manual intervention.
   - It bypasses the need for managing API keys or downloading files manually from Kaggle.

---

### **Explanation of Code**:

1. **Importing KaggleHub**:
   ```python
   import kagglehub
   ```
   - KaggleHub is a Python package designed to streamline the downloading of datasets from Kaggle.

2. **Downloading the Dataset**:
   ```python
   path = kagglehub.dataset_download("franciscoescobar/satellite-images-of-water-bodies")
   ```
   - The `dataset_download` method downloads the dataset by referencing its unique Kaggle identifier (`franciscoescobar/satellite-images-of-water-bodies`).
   - The downloaded dataset is stored in a specific directory, and its path is returned.

3. **Printing the Path**:
   ```python
   print("Path to dataset files:", path)
   ```
   - This ensures that the dataset path is displayed, helping locate the files for subsequent steps like loading and processing.

---

### **Why This Step is Important**:
- **Dataset Accessibility**: Ensures that the required data is available in a usable format.
- **Automated Workflow**: KaggleHub integrates seamlessly with the Python environment, removing the need for manual download or upload steps.

---

### **Outcome**:
After running this step:
- The dataset is successfully downloaded to a local directory.
- Its path is printed for reference in the next steps (e.g., loading images and masks).

## Loading Satellite Images and Masks

---

### **Code**:
```python
from PIL import Image

def load_images_and_masks(dataset_path):
    """
    Loads the satellite images and masks from the dataset path.

    Args:
        dataset_path (str): Path to the dataset root directory.

    Returns:
        image_files (list): List of paths to satellite images.
        mask_files (list): List of paths to corresponding masks.
    """
    images_dir = os.path.join(dataset_path, "Water Bodies Dataset/Images")
    masks_dir = os.path.join(dataset_path, "Water Bodies Dataset/Masks")

    # Load all image and mask files
    image_files = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith(('.jpg', '.tif'))])
    mask_files = sorted([os.path.join(masks_dir, f) for f in os.listdir(masks_dir) if f.endswith(('.jpg', '.tif'))])

    return image_files, mask_files

# Example usage
dataset_path = "/root/.cache/kagglehub/datasets/franciscoescobar/satellite-images-of-water-bodies/versions/2"
image_files, mask_files = load_images_and_masks(dataset_path)
```

---

### **Purpose**:
This step organizes the dataset by separating the satellite images and their corresponding masks into two lists for subsequent processing.

---

### **Explanation of Code**:

1. **Using PIL**:
   ```python
   from PIL import Image
   ```
   - **Why?** The Python Imaging Library (PIL) is used for handling image files, though this function does not manipulate image content directly. It sets up for later steps involving image loading and processing.

2. **Defining the Function**:
   ```python
   def load_images_and_masks(dataset_path):
   ```
   - Encapsulates the logic to separate satellite images and their masks into a reusable function.

3. **Setting Directories**:
   ```python
   images_dir = os.path.join(dataset_path, "Water Bodies Dataset/Images")
   masks_dir = os.path.join(dataset_path, "Water Bodies Dataset/Masks")
   ```
   - Specifies the subdirectories for images and masks within the dataset.

4. **Fetching File Paths**:
   ```python
   image_files = sorted([...])
   mask_files = sorted([...])
   ```
   - **Purpose**: Fetches all file paths with specific extensions (`.jpg`, `.tif`) for both images and masks.
   - **Why `sorted`?** Ensures images and masks are loaded in a consistent order, critical for matching inputs (images) with their ground truth (masks).

5. **Returning Results**:
   ```python
   return image_files, mask_files
   ```
   - Returns two lists: one for image file paths and one for mask file paths.

6. **Example Usage**:
   ```python
   dataset_path = "/root/.cache/kagglehub/datasets/franciscoescobar/satellite-images-of-water-bodies/versions/2"
   image_files, mask_files = load_images_and_masks(dataset_path)
   ```
   - Loads the dataset and assigns the file paths to `image_files` and `mask_files`.

---

### **Why This Step is Important**:
- **Data Organization**: Clearly separates images and their respective masks for further processing.
- **File Matching**: Ensures a one-to-one correspondence between inputs (images) and labels (masks).

---

### **Outcome**:
After running this step:
1. Two lists, `image_files` and `mask_files`, are created.
2. These lists contain paths to all satellite images and their corresponding masks.

## Visualizing Satellite Images and Masks

---

### **Code**:
```python
def visualize_images_and_masks(image_files, mask_files, num_examples=5):
    """
    Visualizes satellite images and their corresponding masks.

    Args:
        image_files (list): List of file paths to satellite images.
        mask_files (list): List of file paths to corresponding masks.
        num_examples (int): Number of examples to display.
    """
    num_examples = min(num_examples, len(image_files), len(mask_files))

    for i in range(num_examples):
        # Load image and mask
        image = Image.open(image_files[i])
        mask = Image.open(mask_files[i])

        # Plot image and mask side by side
        plt.figure(figsize=(10, 5))

        # Display satellite image
        plt.subplot(1, 2, 1)
        plt.title(f"Satellite Image {i+1}")
        plt.imshow(image)
        plt.axis("off")

        # Display corresponding mask
        plt.subplot(1, 2, 2)
        plt.title(f"Mask {i+1}")
        plt.imshow(mask, cmap="gray")
        plt.axis("off")

        plt.show()

# Visualize a few examples
visualize_images_and_masks(image_files, mask_files, num_examples=5)
```

---

### **Purpose**:
This function visually pairs satellite images with their corresponding segmentation masks to ensure:
1. **Data Integrity**: Verifies that images and masks are correctly aligned.
2. **Data Understanding**: Helps in understanding the structure and characteristics of the dataset.

---

### **Explanation of Code**:

1. **Function Definition**:
   ```python
   def visualize_images_and_masks(image_files, mask_files, num_examples=5):
   ```
   - Encapsulates the logic for visualizing image-mask pairs.
   - **Arguments**:
     - `image_files`: List of file paths for satellite images.
     - `mask_files`: List of file paths for corresponding masks.
     - `num_examples`: The number of examples to display (default is 5).

2. **Limiting Number of Examples**:
   ```python
   num_examples = min(num_examples, len(image_files), len(mask_files))
   ```
   - Ensures the number of examples does not exceed the length of the lists.

3. **Loading Images and Masks**:
   ```python
   image = Image.open(image_files[i])
   mask = Image.open(mask_files[i])
   ```
   - **Why PIL?** The Python Imaging Library (Pillow) is used to load image files into a usable format for visualization.

4. **Creating a Plot for Each Pair**:
   ```python
   plt.figure(figsize=(10, 5))
   ```
   - Creates a new plot with specified dimensions for each image-mask pair.

5. **Displaying Satellite Image**:
   ```python
   plt.subplot(1, 2, 1)
   plt.title(f"Satellite Image {i+1}")
   plt.imshow(image)
   plt.axis("off")
   ```
   - Plots the satellite image in the first panel of a 2-panel figure.

6. **Displaying Corresponding Mask**:
   ```python
   plt.subplot(1, 2, 2)
   plt.title(f"Mask {i+1}")
   plt.imshow(mask, cmap="gray")
   plt.axis("off")
   ```
   - Plots the corresponding mask in grayscale in the second panel.

7. **Displaying the Plot**:
   ```python
   plt.show()
   ```
   - Renders the plot, allowing for side-by-side comparison.

---

### **Why This Step is Important**:
1. **Ensures Data Quality**: Identifies potential issues like misaligned masks, corrupted files, or inconsistencies in the dataset.
2. **Visual Validation**: Provides insight into how well masks represent water bodies in satellite images.

---

### **Outcome**:
When this function is executed:
- A specified number of image-mask pairs are displayed side-by-side.
- Allows for manual verification of the dataset before moving to preprocessing and model training.

## Resizing and Preparing Training Images

---

### **Code**:
```python
train_images_array = []

for image_path in image_files:  # Use image_files from Step 3
    # Read and resize the image
    img = cv2.resize(cv2.imread(image_path, cv2.IMREAD_UNCHANGED), (128, 128))
    train_images_array.append(img)

# Convert the list to a NumPy array
train_images_array = np.array(train_images_array, dtype=np.float32)
```

---

### **Purpose**:
This step processes the satellite images by resizing them to a fixed shape and preparing them for model training in a format that the model can efficiently use.

---

### **Explanation of Code**:

1. **Initialize an Empty List**:
   ```python
   train_images_array = []
   ```
   - Prepares a list to store all processed images.

2. **Iterate Over Image Paths**:
   ```python
   for image_path in image_files:
   ```
   - Loops through the file paths of satellite images loaded in the earlier step.

3. **Read and Resize Images**:
   ```python
   img = cv2.resize(cv2.imread(image_path, cv2.IMREAD_UNCHANGED), (128, 128))
   ```
   - **cv2.imread(image_path, cv2.IMREAD_UNCHANGED)**:
     - Reads the image file from the given path.
     - The `cv2.IMREAD_UNCHANGED` flag ensures the image is loaded in its original color depth without modifications.
   - **cv2.resize(image, (128, 128))**:
     - Resizes the image to 128x128 pixels.
     - **Why Resize?** Deep learning models require input tensors of a fixed size, and resizing ensures uniformity across the dataset.

4. **Append Resized Images**:
   ```python
   train_images_array.append(img)
   ```
   - Adds each resized image to the `train_images_array` list.

5. **Convert to NumPy Array**:
   ```python
   train_images_array = np.array(train_images_array, dtype=np.float32)
   ```
   - **Why Convert?**
     - Neural networks process data as multidimensional arrays (tensors).
     - Converting the list to a NumPy array ensures efficient storage and computation.
   - **dtype=np.float32**:
     - Ensures that the data type of pixel values is consistent and suitable for neural network operations.

---

### **Why This Step is Important**:
1. **Prepares Images for Model Input**: Converts images into a consistent size and format required by the neural network.
2. **Improves Computational Efficiency**: Using NumPy arrays optimizes memory usage and speeds up subsequent computations.

---

### **Outcome**:
After running this code:
1. All satellite images are resized to 128x128 pixels.
2. The resized images are stored in `train_images_array` as a NumPy array, ready for preprocessing and model training.

## Resizing and Preparing Mask Images

---

### **Code**:
```python
mask_images_array = []

for mask_path in mask_files:  # Use mask_files from Step 3
    # Read the mask as grayscale and resize it
    msk = cv2.resize(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE), (128, 128))
    mask_images_array.append(msk)

# Convert the list to a NumPy array
mask_images_array = np.array(mask_images_array, dtype=np.uint8)
```

---

### **Purpose**:
This step processes the mask images by resizing them to a fixed shape and converting them into a numerical format suitable for training. Mask images are the ground truth annotations that label water bodies in satellite images.

---

### **Explanation of Code**:

1. **Initialize an Empty List**:
   ```python
   mask_images_array = []
   ```
   - Prepares a list to store all processed mask images.

2. **Iterate Over Mask Paths**:
   ```python
   for mask_path in mask_files:
   ```
   - Loops through the file paths of mask images.

3. **Read and Resize Mask Images**:
   ```python
   msk = cv2.resize(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE), (128, 128))
   ```
   - **cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)**:
     - Reads the mask image file in grayscale mode.
     - Grayscale mode ensures the mask values range from 0 (non-water) to 255 (water), simplifying binary segmentation.
   - **cv2.resize(mask, (128, 128))**:
     - Resizes the mask to 128x128 pixels to match the input image size.

4. **Append Resized Masks**:
   ```python
   mask_images_array.append(msk)
   ```
   - Adds each resized mask to the `mask_images_array` list.

5. **Convert to NumPy Array**:
   ```python
   mask_images_array = np.array(mask_images_array, dtype=np.uint8)
   ```
   - **Why Convert?**
     - Neural networks require input and output in tensor (multi-dimensional array) form.
     - Masks are converted to a NumPy array for efficient storage and processing.
   - **dtype=np.uint8**:
     - Ensures that mask values are stored as integers between 0 and 255, conserving memory.

---

### **Why This Step is Important**:
1. **Prepares Masks for Model Output**: Resizes masks to match the input image size, ensuring proper alignment during training.
2. **Simplifies Mask Processing**: Converting masks to grayscale and using integer values makes it easier to interpret water vs. non-water pixels.

---

### **Outcome**:
After running this code:
1. All mask images are resized to 128x128 pixels.
2. The resized masks are stored in `mask_images_array` as a NumPy array, ready for preprocessing and model training.

## Checking the Length of the Training Images Array

---

### **Code**:
```python
len(train_images_array)
```

---

### **Purpose**:
This line checks the number of training images stored in the `train_images_array`. It helps ensure that all images have been successfully processed and loaded into the array.

---

### **Explanation**:
1. **`train_images_array`**:
   - This array contains all the resized satellite images prepared in the previous step.
   - It was created by resizing each image to 128x128 pixels and storing them as a NumPy array.

2. **Using `len()`**:
   - The `len()` function returns the number of elements in a list, tuple, or array.
   - In this case, it returns the total count of images loaded into `train_images_array`.

---

### **Why This Step is Important**:
1. **Verification**:
   - Confirms that all images from `image_files` have been successfully loaded and processed.
   - Helps identify any discrepancies or missing images before proceeding.

2. **Debugging**:
   - If the length is unexpectedly low, it could indicate issues such as:
     - Incorrect file paths.
     - Missing or corrupted image files.
     - Errors during resizing or loading.

---

### **Outcome**:
After running this code:
- The total number of training images is displayed.
- You can compare this number with the length of `image_files` to ensure consistency.

## Checking the Length of the Mask Images Array

---

### **Code**:
```python
len(mask_images_array)
```

---

### **Purpose**:
This line checks the number of mask images stored in the `mask_images_array`. It ensures that all corresponding ground truth masks have been successfully processed and loaded into the array.

---

### **Explanation**:
1. **`mask_images_array`**:
   - This array contains all the resized mask images (binary segmentation maps) prepared in the previous step.
   - Each mask corresponds to a satellite image in `train_images_array`.

2. **Using `len()`**:
   - The `len()` function returns the number of elements in the array.
   - In this context, it outputs the total count of masks in `mask_images_array`.

---

### **Why This Step is Important**:
1. **Verification**:
   - Ensures all masks from `mask_files` have been correctly loaded and processed.
   - The length of `mask_images_array` should match the length of `train_images_array` since each image must have a corresponding mask.

2. **Debugging**:
   - A mismatch in the lengths of `train_images_array` and `mask_images_array` could indicate:
     - Missing or misaligned mask files.
     - Errors during mask loading or resizing.

---

### **Outcome**:
After running this code:
- The total number of processed mask images is displayed.
- You can compare this number with the length of `mask_files` or `train_images_array` to confirm consistency.

## Step Explanation: Checking the Shape of a Training Image

---

### **Code**:
```python
train_images_array[0].shape
```

---

### **Purpose**:
This line retrieves the shape of the first training image in the `train_images_array`. It confirms that each image has been resized correctly and verifies its dimensions.

---

### **Explanation**:
1. **`train_images_array[0]`**:
   - Accesses the first image in the `train_images_array`, which is a NumPy array storing all the resized images.

2. **`.shape`**:
   - The `.shape` attribute provides the dimensions of the selected image as a tuple.
   - For an RGB image resized to 128x128, the shape should be `(128, 128, 3)`:
     - **128, 128**: Height and width of the image.
     - **3**: Number of color channels (Red, Green, Blue).

---

### **Why This Step is Important**:
1. **Verification**:
   - Ensures all images have been resized to the correct dimensions (128x128 pixels).
   - Confirms the presence of 3 color channels (RGB).

2. **Debugging**:
   - If the shape does not match the expected `(128, 128, 3)`:
     - The image may not have been resized correctly.
     - The file may not be in the expected RGB format.

---

### **Outcome**:
After running this code:
- The shape of the first training image is displayed.
- The shape should typically be `(128, 128, 3)` for RGB images resized to 128x128.

## Step Explanation: Checking the Shape of a Mask Image

---

### **Code**:
```python
mask_images_array[0].shape
```

---

### **Purpose**:
This line retrieves the shape of the first mask image in the `mask_images_array`. It ensures that each mask has been resized correctly and verifies its dimensions.

---

### **Explanation**:
1. **`mask_images_array[0]`**:
   - Accesses the first mask in the `mask_images_array`, which stores all the resized binary masks as NumPy arrays.

2. **`.shape`**:
   - The `.shape` attribute provides the dimensions of the selected mask as a tuple.
   - For a grayscale mask resized to 128x128, the shape should be `(128, 128)`:
     - **128, 128**: Height and width of the mask.
     - Masks are stored as 2D arrays because they are grayscale (no color channels).

---

### **Why This Step is Important**:
1. **Verification**:
   - Confirms that masks have been resized correctly to 128x128 pixels.
   - Ensures the masks are in the expected 2D format (grayscale).

2. **Debugging**:
   - If the shape does not match the expected `(128, 128)`:
     - The mask may not have been resized correctly.
     - The file may not be in the expected grayscale format.

---

### **Outcome**:
After running this code:
- The shape of the first mask image is displayed.
- The shape should typically be `(128, 128)` for grayscale masks resized to 128x128.

## Step Explanation: Normalizing Images and Masks

---

### **Code**:
```python
def normalize_array(arr):
    return arr / 255.0

X = normalize_array(train_images_array)
y = normalize_array(mask_images_array)
```

---

### **Purpose**:
This step normalizes the pixel values of the training images and masks. Normalization is a crucial preprocessing step to scale the data into a range that enhances model training efficiency.

---

### **Explanation**:

1. **Define a Normalization Function**:
   ```python
   def normalize_array(arr):
       return arr / 255.0
   ```
   - **Input**: Takes a NumPy array (`arr`) as input.
   - **Output**: Divides each element of the array by `255.0` to scale pixel values into the range `[0, 1]`.
     - **Why 255?**:
       - Pixel values in images typically range from `0` to `255`.
       - Dividing by `255.0` converts these values into the normalized range `[0, 1]`.

2. **Normalize Training Images**:
   ```python
   X = normalize_array(train_images_array)
   ```
   - Scales the RGB values of the satellite images to `[0, 1]`.
   - Ensures uniformity in pixel intensity, reducing bias caused by varying lighting or pixel scales.

3. **Normalize Masks**:
   ```python
   y = normalize_array(mask_images_array)
   ```
   - Scales the mask pixel values (0 or 255 for binary segmentation) to `0` or `1`.
   - Simplifies binary classification tasks by directly representing water (1) and non-water (0).

---

### **Why Normalization is Important**:
1. **Improved Training Stability**:
   - Neural networks perform better when input values are small and within a consistent range.
   - Reduces the risk of exploding or vanishing gradients.

2. **Faster Convergence**:
   - Normalization allows the optimizer to converge more quickly by avoiding large fluctuations in gradient updates.

3. **Ensures Consistency**:
   - Both input images (`X`) and ground truth masks (`y`) are scaled similarly, maintaining the same range for model input and output.

---

### **Outcome**:
After running this step:
1. `X` contains normalized training images with pixel values in `[0, 1]`.
2. `y` contains normalized masks with pixel values either `0` (non-water) or `1` (water).
3. The normalized arrays are ready for further splitting into training and testing datasets.

## Step Explanation: Splitting Data into Training and Testing Sets

---

### **Code**:
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

### **Purpose**:
This step splits the dataset into training and testing sets. It ensures that the model is trained on one portion of the data and evaluated on another to test its generalization capabilities.

---

### **Explanation**:

1. **Importing `train_test_split`**:
   ```python
   from sklearn.model_selection import train_test_split
   ```
   - **What It Does**: A function from Scikit-learn that splits datasets into random subsets for training and testing.
   - **Why Use It?**:
     - Simplifies the process of creating training and testing datasets.
     - Ensures reproducibility with options like `random_state`.

2. **Splitting the Data**:
   ```python
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   ```
   - **Arguments**:
     - `X`: Normalized satellite images (input features).
     - `y`: Normalized masks (output labels).
     - `test_size=0.2`: Specifies that 20% of the data will be used for testing, while 80% will be used for training.
     - `random_state=42`: Ensures the split is reproducible (the same subsets are created each time the code runs).
   - **Outputs**:
     - `X_train`: Training images (80% of `X`).
     - `X_test`: Testing images (20% of `X`).
     - `y_train`: Training masks corresponding to `X_train`.
     - `y_test`: Testing masks corresponding to `X_test`.

---

### **Why This Step is Important**:
1. **Ensures Generalization**:
   - Splitting the data into training and testing subsets ensures the model is evaluated on unseen data, reflecting its ability to generalize.

2. **Prevents Overfitting**:
   - By withholding a portion of the data during training, the model's performance is tested on data it hasn’t encountered, reducing the risk of overfitting.

3. **Balanced Evaluation**:
   - The `test_size=0.2` ensures a balanced division of the dataset, leaving enough data for both training and testing.

---

### **Outcome**:
After running this code:
1. `X_train` and `y_train` contain the training data for model fitting.
2. `X_test` and `y_test` contain the testing data for evaluating the model’s performance.
3. The data is now ready for model training.

## Defining a Convolutional Block

---

### **Code**:
```python
def conv_block(inputs=None, n_filters=32, dropout_prob=0, max_pooling=True):

    conv = Conv2D(n_filters,  # Number of filters
                  3,  # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='he_normal')(inputs)
    conv = Conv2D(n_filters,  # Number of filters
                  3,  # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='he_normal')(conv)

    if dropout_prob > 0:
        conv = Dropout(dropout_prob)(conv)

    if max_pooling:
        next_layer = MaxPooling2D(pool_size=(2, 2))(conv)
    else:
        next_layer = conv

    skip_connection = conv

    return next_layer, skip_connection
```

---

### **Purpose**:
This function defines a **convolutional block**, a fundamental building block of the U-Net architecture. It extracts features from the input image, optionally applies dropout for regularization, and reduces spatial dimensions using max pooling if specified.

---

### **Explanation**:

1. **Inputs**:
   - `inputs`: The input tensor to the block.
   - `n_filters=32`: Number of convolutional filters (default is 32).
   - `dropout_prob=0`: Dropout probability to randomly deactivate neurons for regularization (default is 0, meaning no dropout).
   - `max_pooling=True`: Whether to apply max pooling for downsampling (default is `True`).

2. **Convolutional Layers**:
   ```python
   conv = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
   ```
   - **Conv2D**:
     - Applies a 2D convolution operation with:
       - `n_filters`: Number of filters (depth of output tensor).
       - `3`: Kernel size (3x3 convolution filter).
       - `activation='relu'`: Activation function to introduce non-linearity.
       - `padding='same'`: Ensures the output has the same spatial dimensions as the input.
       - `kernel_initializer='he_normal'`: Initializes weights for faster convergence.
   - Two convolutional layers are applied sequentially to enhance feature extraction.

3. **Dropout (Optional)**:
   ```python
   if dropout_prob > 0:
       conv = Dropout(dropout_prob)(conv)
   ```
   - **Purpose**: Reduces overfitting by randomly setting a fraction of neurons to zero during training.
   - Only applied if `dropout_prob` is greater than 0.

4. **Max Pooling (Optional)**:
   ```python
   if max_pooling:
       next_layer = MaxPooling2D(pool_size=(2, 2))(conv)
   ```
   - **Purpose**: Reduces the spatial dimensions (height and width) of the feature maps by taking the maximum value in each 2x2 window.
   - If `max_pooling=False`, the feature map is passed unchanged.

5. **Skip Connection**:
   ```python
   skip_connection = conv
   ```
   - Stores the output of the convolutional layers before pooling.
   - **Why?** Skip connections are used in U-Net to merge encoder features with decoder features, preserving spatial details.

6. **Return Values**:
   ```python
   return next_layer, skip_connection
   ```
   - `next_layer`: The processed feature map to be passed to the next layer.
   - `skip_connection`: The intermediate feature map for later use in the decoder.

---

### **Why This Step is Important**:
1. **Feature Extraction**:
   - Convolutional layers detect patterns like edges, textures, and shapes in the input data.
   - Stacking multiple convolutional layers enhances feature extraction at different levels.

2. **Downsampling**:
   - Max pooling reduces spatial dimensions while retaining important features, enabling the model to learn abstract representations efficiently.

3. **Regularization**:
   - Dropout helps mitigate overfitting, especially in complex models with many parameters.

4. **Skip Connections**:
   - A hallmark of U-Net, these connections combine encoder and decoder features, improving the model's ability to produce accurate segmentations.

---

### **Outcome**:
After defining this function:
- It can be used repeatedly in the U-Net architecture to build encoder layers.
- Outputs both the downsampled feature map (`next_layer`) and the intermediate feature map (`skip_connection`) for later use.

## Defining an Upsampling Block

---

### **Code**:
```python
def upsampling_block(expansive_input, contractive_input, n_filters=32):

    up = Conv2DTranspose(
                 n_filters,    # number of filters
                 3,    # Kernel size
                 strides=(2, 2),
                 padding='same')(expansive_input)

    merge = concatenate([up, contractive_input], axis=3)
    conv = Conv2D(n_filters,   # Number of filters
                 3,     # Kernel size
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(merge)
    conv = Conv2D(n_filters,  # Number of filters
                 3,   # Kernel size
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(conv)

    return conv
```

---

### **Purpose**:
The **upsampling block** is a fundamental component of the U-Net's decoder. It restores spatial dimensions by upsampling feature maps and combines these features with those from the encoder via skip connections. This process helps refine segmentation predictions.

---

### **Explanation**:

1. **Inputs**:
   - `expansive_input`: The input feature map from the previous layer of the decoder.
   - `contractive_input`: The feature map from the encoder (skip connection) at the same spatial level.
   - `n_filters=32`: Number of filters for the convolutional layers (default is 32).

---

### **Steps in the Block**:

#### 1. **Transposed Convolution (Upsampling)**:
```python
up = Conv2DTranspose(
                 n_filters,
                 3,
                 strides=(2, 2),
                 padding='same')(expansive_input)
```
- **Conv2DTranspose**:
  - Performs transposed convolution (also called deconvolution or up-convolution) to upsample the feature map.
  - **Arguments**:
    - `n_filters`: Number of filters, determining the depth of the output feature map.
    - `3`: Kernel size (3x3 filter).
    - `strides=(2, 2)`: Upsamples the feature map by a factor of 2 in both height and width.
    - `padding='same'`: Ensures the output has the same spatial dimensions as the input after upsampling.
- **Purpose**: Doubles the spatial dimensions of the feature map, restoring resolution lost during downsampling.

---

#### 2. **Concatenation (Skip Connection)**:
```python
merge = concatenate([up, contractive_input], axis=3)
```
- Combines the upsampled feature map (`up`) with the corresponding encoder feature map (`contractive_input`).
- **Axis=3**: Combines along the channel dimension.
- **Why?**
  - Skip connections help preserve fine-grained spatial details from the encoder that might be lost during downsampling.

---

#### 3. **Convolutional Layers**:
```python
conv = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge)
conv = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv)
```
- Two convolutional layers are applied to the concatenated feature map:
  - **n_filters**: Number of filters in each layer, defining the depth of the feature map.
  - **3**: Kernel size (3x3 filter).
  - **activation='relu'**: Introduces non-linearity.
  - **padding='same'**: Maintains spatial dimensions.
  - **kernel_initializer='he_normal'**: Efficient weight initialization.
- **Purpose**:
  - Refines features after upsampling.
  - Enhances the representation for the target segmentation task.

---

### **Output**:
```python
return conv
```
- Returns the refined feature map after upsampling and concatenation.

---

### **Why This Step is Important**:
1. **Restores Spatial Resolution**:
   - Transposed convolution increases the spatial dimensions, reconstructing the original input resolution step by step.

2. **Refines Segmentation Predictions**:
   - Combining decoder features with encoder features ensures that both high-level semantic and low-level spatial information are used.

3. **Preserves Context**:
   - Skip connections maintain critical details that may otherwise be lost during encoding.

---

### **Outcome**:
After defining this function:
- It can be repeatedly used in the decoder part of the U-Net architecture to build layers that upsample feature maps and merge them with encoder features.
- This process is crucial for maintaining spatial resolution and refining segmentation predictions in image segmentation tasks.

## Building the U-Net Model

---

### **Code**:
```python
def unet_model(input_size=(128, 128, 3), n_filters=32):

    inputs = Input(input_size)

    # Encoder blocks
    cblock1 = conv_block(inputs, n_filters)
    cblock2 = conv_block(cblock1[0], 2*n_filters)
    cblock3 = conv_block(cblock2[0], 2**2*n_filters)
    cblock4 = conv_block(cblock3[0], 2**3*n_filters, dropout_prob=0.3)
    cblock5 = conv_block(cblock4[0], 2**4*n_filters, dropout_prob=0.3, max_pooling=False)

    # Decoder blocks
    ublock6 = upsampling_block(cblock5[0], cblock4[1],  2**3*n_filters)
    ublock7 = upsampling_block(ublock6, cblock3[1],  2**2*n_filters)
    ublock8 = upsampling_block(ublock7, cblock2[1],  2*n_filters)
    ublock9 = upsampling_block(ublock8, cblock1[1],  n_filters)

    # Final convolution
    conv9 = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(ublock9)
    conv10 = Conv2D(1, 1, padding='same', activation='sigmoid')(conv9)

    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    return model
```

---

### **Purpose**:
This function builds the **U-Net architecture**, a deep convolutional neural network for semantic segmentation tasks. It takes in an image, processes it through an encoder-decoder architecture, and outputs a segmentation mask.

---

### **Explanation**:

#### 1. **Inputs**:
```python
inputs = Input(input_size)
```
- **Purpose**: Defines the input layer of the model with the specified shape (`input_size`), e.g., `(128, 128, 3)` for 128x128 RGB images.

---

#### 2. **Encoder Blocks**:
```python
cblock1 = conv_block(inputs, n_filters)
cblock2 = conv_block(cblock1[0], 2*n_filters)
cblock3 = conv_block(cblock2[0], 2**2*n_filters)
cblock4 = conv_block(cblock3[0], 2**3*n_filters, dropout_prob=0.3)
cblock5 = conv_block(cblock4[0], 2**4*n_filters, dropout_prob=0.3, max_pooling=False)
```
- **Purpose**: Extracts features from the input using successive convolutional blocks.
- **Process**:
  - Each block applies convolutions, optional dropout, and max pooling (except for `cblock5` where pooling is disabled).
  - The number of filters (`n_filters`) increases with depth, capturing progressively higher-level features.
- **Outputs**:
  - Each block returns:
    - `next_layer`: Downsampled feature map.
    - `skip_connection`: Features for use in the decoder via skip connections.

---

#### 3. **Decoder Blocks**:
```python
ublock6 = upsampling_block(cblock5[0], cblock4[1],  2**3*n_filters)
ublock7 = upsampling_block(ublock6, cblock3[1],  2**2*n_filters)
ublock8 = upsampling_block(ublock7, cblock2[1],  2*n_filters)
ublock9 = upsampling_block(ublock8, cblock1[1],  n_filters)
```
- **Purpose**: Restores the spatial resolution of feature maps to match the input size, refining predictions at each step.
- **Process**:
  - Each block upsamples the feature map from the previous decoder layer.
  - Combines the upsampled map with the corresponding encoder feature map using skip connections.
  - Refines features through convolutions.

---

#### 4. **Final Convolutional Layers**:
```python
conv9 = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(ublock9)
conv10 = Conv2D(1, 1, padding='same', activation='sigmoid')(conv9)
```
- **conv9**:
  - A convolutional layer with `n_filters` to refine the final decoder features.
- **conv10**:
  - Outputs a single-channel prediction (segmentation mask).
  - **Sigmoid Activation**:
    - Produces values in the range [0, 1], suitable for binary segmentation tasks (e.g., water vs. non-water).

---

#### 5. **Model Definition**:
```python
model = tf.keras.Model(inputs=inputs, outputs=conv10)
```
- Combines all layers into a functional model with:
  - `inputs`: The input layer.
  - `outputs`: The final segmentation mask.

---

### **Why This Step is Important**:
1. **Feature Hierarchy**:
   - The encoder captures both low-level and high-level features.
2. **Spatial Precision**:
   - The decoder restores spatial resolution, integrating details from skip connections.
3. **Binary Segmentation**:
   - The final output produces a probability map for each pixel, indicating whether it belongs to the target class (e.g., water).

---

### **Outcome**:
After running this function:
1. A U-Net model is built and returned as a TensorFlow `Model` object.
2. The model is ready for compilation and training.

## Compiling the U-Net Model

---

### **Code**:
```python
model = unet_model(n_filters=32, input_size=(128, 128, 3))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

---

### **Purpose**:
This step builds the U-Net model using the previously defined `unet_model` function and compiles it for training. Compilation configures the model's optimizer, loss function, and evaluation metrics.

---

### **Explanation**:

#### 1. **Building the Model**:
```python
model = unet_model(n_filters=32, input_size=(128, 128, 3))
```
- **Purpose**: Calls the `unet_model` function to create a U-Net architecture.
- **Arguments**:
  - `n_filters=32`: Sets the number of filters in the initial convolutional layers (subsequent layers will have multiples of this value).
  - `input_size=(128, 128, 3)`:
    - Defines the input shape of the model as 128x128 RGB images.
    - **Why 3?** Each image has 3 color channels (Red, Green, Blue).

---

#### 2. **Compiling the Model**:
```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```
- **Purpose**: Configures the learning process of the model by specifying:
  1. **Optimizer**:
     ```python
     optimizer='adam'
     ```
     - **Adam Optimizer**:
       - Combines the advantages of RMSProp and momentum optimizers.
       - Adapts the learning rate for each parameter, enabling efficient training and faster convergence.
       - Suitable for segmentation tasks involving large datasets.
  2. **Loss Function**:
     ```python
     loss='binary_crossentropy'
     ```
     - **Binary Crossentropy**:
       - Measures the difference between predicted probabilities and true binary labels (e.g., water vs. non-water).
       - Formula:
         \[
         \text{Loss} = - \frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
         \]
       - Encourages the model to output probabilities close to 1 for the target class (water) and 0 for the background.
  3. **Metrics**:
     ```python
     metrics=['accuracy']
     ```
     - **Accuracy**:
       - Measures the percentage of correctly classified pixels.
       - While accuracy is straightforward, it may not fully capture the performance of a segmentation model (e.g., IoU is often preferred for evaluation).

---

### **Why This Step is Important**:
1. **Defines Learning Behavior**:
   - The optimizer, loss function, and metrics determine how the model updates weights during training and evaluates performance.
2. **Prepares the Model for Training**:
   - Compilation is a prerequisite for calling `model.fit()` to start training.

---

### **Outcome**:
After this step:
1. The U-Net model is built and ready for training.
2. The learning process is configured with Adam optimization, binary crossentropy loss, and accuracy evaluation.

## Summarizing the U-Net Model

---

### **Code**:
```python
model.summary()
```

---

### **Purpose**:
The `summary()` method provides a detailed overview of the U-Net model's architecture, including information about each layer, the number of parameters, and the input/output shapes at each stage.

---

### **Explanation**:

1. **Model Architecture Details**:
   - `model.summary()` outputs a table with the following details for each layer:
     - **Layer Name**: Name of each layer in the model.
     - **Layer Type**: Type of operation (e.g., `Conv2D`, `MaxPooling2D`, `Conv2DTranspose`).
     - **Output Shape**: Dimensions of the output tensor for that layer.
     - **Number of Parameters**: Total number of learnable parameters (weights and biases).

2. **Global Information**:
   - The summary includes:
     - **Total Parameters**: Total number of trainable parameters in the model.
     - **Trainable Parameters**: Parameters that are updated during training.
     - **Non-trainable Parameters**: Parameters fixed during training (e.g., those in frozen layers).

---

### **Why This Step is Important**:

1. **Verifies Model Design**:
   - Confirms the correctness of the architecture (e.g., input and output dimensions match the expected values).

2. **Resource Estimation**:
   - Understanding the number of parameters helps estimate the computational and memory resources required for training.

3. **Debugging**:
   - Detects potential issues like mismatched shapes or unexpected layers.

---

### **Example Output for a U-Net Model**:
```plaintext
Layer (type)                   Output Shape              Param #
=================================================================
input_1 (InputLayer)           [(None, 128, 128, 3)]     0
_________________________________________________________________
conv2d (Conv2D)                (None, 128, 128, 32)      896
_________________________________________________________________
conv2d_1 (Conv2D)              (None, 128, 128, 32)      9248
_________________________________________________________________
max_pooling2d (MaxPooling2D)   (None, 64, 64, 32)        0
_________________________________________________________________
...
conv2d_transpose (Conv2DTransp (None, 128, 128, 32)      9248
_________________________________________________________________
concatenate (Concatenate)      (None, 128, 128, 64)      0
_________________________________________________________________
conv2d_final (Conv2D)          (None, 128, 128, 1)       577
=================================================================
Total params: 1,941,889
Trainable params: 1,941,889
Non-trainable params: 0
```

---

### **Outcome**:
After running this step:
1. You receive a detailed summary of the U-Net model’s architecture.
2. It helps verify the design and assess the complexity of the model.

## Step Explanation: Defining Callbacks for Training

---

### **Code**:
```python
# Early stopping callback to stop training when validation loss stops improving
early_stopping_cb = EarlyStopping(patience=5, restore_best_weights=True)

# Model checkpoint callback to save the best model during training
model_checkpoint_cb = ModelCheckpoint(
    filepath="model.keras",  # Changed the file path to end with .keras
    save_best_only=True,  # Only save the model if the validation loss improves
    monitor="val_loss",   # Monitor validation loss for improvement
    mode="min"            # Minimize the monitored metric (validation loss)
)
```

---

### **Purpose**:
Callbacks are used during training to automate certain tasks, such as saving the best model and stopping training when further improvement is unlikely. This step defines two callbacks:
1. **EarlyStopping**: Halts training when the validation loss stops improving.
2. **ModelCheckpoint**: Saves the model with the best validation performance.

---

### **Explanation**:

#### 1. **EarlyStopping Callback**:
```python
early_stopping_cb = EarlyStopping(patience=5, restore_best_weights=True)
```
- **Purpose**:
  - Monitors a specific metric (`val_loss` by default).
  - Stops training if the metric does not improve for a specified number of epochs (`patience`).
- **Arguments**:
  - `patience=5`: Training stops if there is no improvement in validation loss for 5 consecutive epochs.
  - `restore_best_weights=True`: Ensures that the model's weights are reverted to the best-performing state when training ends.

#### 2. **ModelCheckpoint Callback**:
```python
model_checkpoint_cb = ModelCheckpoint(
    filepath="model.keras",
    save_best_only=True,
    monitor="val_loss",
    mode="min"
)
```
- **Purpose**:
  - Saves the model whenever validation loss improves.
  - Prevents losing the best-performing model during training.
- **Arguments**:
  - `filepath="model.keras"`: Specifies the filename for saving the model.
  - `save_best_only=True`: Ensures only the best-performing model (lowest `val_loss`) is saved.
  - `monitor="val_loss"`: Tracks the validation loss as the metric for improvement.
  - `mode="min"`: Indicates that lower values of `val_loss` are better.

---

### **Why These Callbacks Are Important**:

#### **1. EarlyStopping**:
- **Prevents Overfitting**: Stops training when the model starts overfitting (validation loss stagnates or increases).
- **Saves Resources**: Avoids wasting time and computation on unnecessary epochs.

#### **2. ModelCheckpoint**:
- **Saves the Best Model**: Ensures that the best-performing model during training is preserved, even if later epochs degrade performance.
- **Simplifies Deployment**: The saved model can be directly loaded and used without additional processing.

---

### **Usage in Training**:
```python
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    callbacks=[early_stopping_cb, model_checkpoint_cb]
)
```
- **callbacks**:
  - Includes the `early_stopping_cb` and `model_checkpoint_cb` to monitor and save progress during training.

---

### **Outcome**:
After defining these callbacks:
1. Training stops early if validation performance does not improve.
2. The best-performing model is automatically saved for future use.

## Training the U-Net Model

---

### **Code**:
```python
num_epochs = 50

history = model.fit(X_train, y_train, epochs=num_epochs, callbacks=[early_stopping_cb], validation_data=(X_test, y_test))
```

---

### **Purpose**:
This step trains the U-Net model on the prepared dataset (`X_train` and `y_train`) while evaluating its performance on the validation set (`X_test` and `y_test`). It leverages the `EarlyStopping` callback to halt training if validation loss stops improving.

---

### **Explanation**:

#### 1. **Number of Epochs**:
```python
num_epochs = 50
```
- **Purpose**: Specifies the maximum number of complete passes through the training dataset.
- **Why 50?**
  - A sufficiently large number ensures the model has enough opportunities to learn patterns in the data.
  - Training may terminate early due to the `EarlyStopping` callback if validation loss stabilizes or worsens.

---

#### 2. **Model Training (`model.fit`)**:
```python
history = model.fit(X_train, y_train, epochs=num_epochs, callbacks=[early_stopping_cb], validation_data=(X_test, y_test))
```
- **Arguments**:
  - `X_train, y_train`: The training data (images and corresponding masks).
  - `epochs=num_epochs`: The maximum number of training iterations.
  - `callbacks=[early_stopping_cb]`: Includes the `EarlyStopping` callback to monitor validation loss and stop training early if improvement ceases.
  - `validation_data=(X_test, y_test)`:
    - A separate validation set to evaluate the model's performance after each epoch.
    - Used to monitor `val_loss` for early stopping.
- **Output**:
  - `history`: An object that stores training progress, including:
    - Loss and accuracy for both training and validation sets at each epoch.

---

### **Why This Step is Important**:

1. **Optimizes the Model**:
   - Repeated updates to the model’s weights during training minimize the loss function, improving segmentation accuracy.

2. **Monitors Generalization**:
   - Validation data provides insight into how well the model generalizes to unseen data, helping prevent overfitting.

3. **Early Stopping Saves Resources**:
   - Training stops automatically when further improvement in validation loss is unlikely, saving computation time.

---

### **Outcome**:
After this step:
1. The model is trained on the training set (`X_train`, `y_train`) for up to 50 epochs or until early stopping.
2. The training history (`history`) contains metrics for each epoch, which can be visualized to analyze performance trends.

## Visualizing Accuracy During Training

---

### **Code**:
```python
# Creating an accuracy graph for training and testing data
plt.plot(history.history['accuracy'], color='blue', label='Training Accuracy')
plt.plot(history.history['val_accuracy'], color='red', label='Testing Accuracy')
plt.legend()
plt.show()
```

---

### **Purpose**:
This code visualizes the accuracy of the U-Net model on the training and testing datasets during training. It helps identify trends, such as improving accuracy, overfitting, or underfitting.

---

### **Explanation**:

1. **Plotting Training Accuracy**:
   ```python
   plt.plot(history.history['accuracy'], color='blue', label='Training Accuracy')
   ```
   - Accesses the `accuracy` metric from the `history` object, which stores the training accuracy values for each epoch.
   - Plots the training accuracy values over the number of epochs.
   - **Color**: Blue line represents training accuracy.
   - **Label**: Identifies this line in the legend as "Training Accuracy."

2. **Plotting Validation Accuracy**:
   ```python
   plt.plot(history.history['val_accuracy'], color='red', label='Testing Accuracy')
   ```
   - Accesses the `val_accuracy` metric from the `history` object, which stores the validation (testing) accuracy values for each epoch.
   - Plots the validation accuracy values over the number of epochs.
   - **Color**: Red line represents testing accuracy.
   - **Label**: Identifies this line in the legend as "Testing Accuracy."

3. **Adding a Legend**:
   ```python
   plt.legend()
   ```
   - Displays a legend to differentiate between the training and testing accuracy curves.

4. **Displaying the Plot**:
   ```python
   plt.show()
   ```
   - Renders the accuracy graph.

---

### **Why This Step is Important**:
1. **Performance Tracking**:
   - Allows you to observe how the model’s accuracy improves over epochs.
   - A consistent upward trend indicates effective learning.

2. **Identifies Overfitting or Underfitting**:
   - **Overfitting**: Training accuracy is much higher than validation accuracy, indicating the model performs poorly on unseen data.
   - **Underfitting**: Both training and validation accuracies remain low, suggesting the model is not learning effectively.

3. **Visual Insight**:
   - Provides an intuitive understanding of model performance during training, complementing numerical evaluation metrics.

---

### **Outcome**:
After running this code:
1. A line graph is displayed, showing training and testing accuracy for each epoch.
2. You can analyze the model’s learning behavior over time and make adjustments if necessary.

## Visualizing Loss During Training

---

### **Code**:
```python
# Creating a loss graph for training and testing data
plt.plot(history.history['loss'], color='blue', label='Training Loss')
plt.plot(history.history['val_loss'], color='red', label='Testing Loss')
plt.legend()
plt.show()
```

---

### **Purpose**:
This code visualizes the loss of the U-Net model on the training and testing datasets during training. The loss curve helps assess how well the model minimizes the error over time.

---

### **Explanation**:

1. **Plotting Training Loss**:
   ```python
   plt.plot(history.history['loss'], color='blue', label='Training Loss')
   ```
   - Accesses the `loss` values from the `history` object, which stores the training loss for each epoch.
   - Plots the training loss values over the number of epochs.
   - **Color**: Blue line represents the training loss.
   - **Label**: Identifies this line in the legend as "Training Loss."

2. **Plotting Validation Loss**:
   ```python
   plt.plot(history.history['val_loss'], color='red', label='Testing Loss')
   ```
   - Accesses the `val_loss` values from the `history` object, which stores the validation (testing) loss for each epoch.
   - Plots the validation loss values over the number of epochs.
   - **Color**: Red line represents the validation loss.
   - **Label**: Identifies this line in the legend as "Testing Loss."

3. **Adding a Legend**:
   ```python
   plt.legend()
   ```
   - Displays a legend to differentiate between the training and testing loss curves.

4. **Displaying the Plot**:
   ```python
   plt.show()
   ```
   - Renders the loss graph.

---

### **Why This Step is Important**:
1. **Tracks Model Training**:
   - Observes how the model minimizes its error (loss) over epochs.
   - A steady decline in loss indicates successful optimization.

2. **Detects Overfitting or Underfitting**:
   - **Overfitting**: Training loss decreases, but validation loss stagnates or increases, suggesting poor generalization to unseen data.
   - **Underfitting**: Both training and validation losses remain high, indicating the model struggles to learn from the data.

3. **Validation of Early Stopping**:
   - Confirms whether early stopping was triggered at the right point (e.g., when validation loss plateaued or worsened).

---

### **Outcome**:
After running this code:
1. A line graph is displayed, showing the training and testing loss for each epoch.
2. The graph helps evaluate the model’s learning behavior and the effectiveness of early stopping.

## Visualizing Predictions

---

### **Code**:
```python
def visualize_predictions(image_files, mask_files, model, num_examples=5, img_size=(128, 128)):
    """
    Visualizes randomly selected satellite images, ground truth masks, and predicted masks.

    Args:
        image_files (list): List of file paths to satellite images.
        mask_files (list): List of file paths to corresponding masks.
        model: Trained model for predicting masks.
        num_examples (int): Number of examples to display.
        img_size (tuple): Dimensions of predicted masks (height, width).
    """
    num_examples = min(num_examples, len(image_files), len(mask_files))

    # Randomly select indices
    random_indices = random.sample(range(len(image_files)), num_examples)

    for i, idx in enumerate(random_indices):
        # Load the original image and mask
        image = Image.open(image_files[idx])
        mask = Image.open(mask_files[idx])

        # Preprocess image for prediction
        input_img = np.array(image)  # Convert image to NumPy array

        # Resize the image to match the model's input shape
        input_img = cv2.resize(input_img, img_size)  # Resize using OpenCV

        input_img = np.expand_dims(input_img, axis=0)  # Add batch dimension

        # Predict the mask
        predicted_mask = model.predict(input_img).reshape(img_size)

        # Plot image, ground truth mask, and predicted mask
        plt.figure(figsize=(15, 5))

        # Display original image
        plt.subplot(1, 3, 1)
        plt.title(f"Original Image {i+1}")
        plt.imshow(image)
        plt.axis("off")

        # Display ground truth mask
        plt.subplot(1, 3, 2)
        plt.title(f"Ground Truth Mask {i+1}")
        plt.imshow(mask, cmap="gray")
        plt.axis("off")

        # Display predicted mask
        plt.subplot(1, 3, 3)
        plt.title(f"Predicted Mask {i+1}")
        plt.imshow(predicted_mask, cmap="gray")
        plt.axis("off")

        plt.show()

# Example usage
visualize_predictions(image_files, mask_files, model, num_examples=3, img_size=(128, 128))
```

---

### **Purpose**:
This function visually compares the original satellite images, their corresponding ground truth masks, and the predicted masks generated by the trained U-Net model. It helps evaluate the model's segmentation performance qualitatively.

---

### **Explanation**:

1. **Function Definition**:
   - **`image_files`**: List of file paths for satellite images.
   - **`mask_files`**: List of file paths for ground truth masks.
   - **`model`**: Trained U-Net model used for prediction.
   - **`num_examples`**: Number of examples to visualize (default is 5).
   - **`img_size`**: Dimensions to resize the input and output masks (default is `(128, 128)`).

---

### **Steps in the Function**:

#### 1. **Randomly Select Examples**:
```python
random_indices = random.sample(range(len(image_files)), num_examples)
```
- Randomly selects a specified number of image-mask pairs for visualization.
- Ensures variety in the examples displayed.

---

#### 2. **Load and Preprocess Images**:
```python
image = Image.open(image_files[idx])
mask = Image.open(mask_files[idx])
input_img = np.array(image)
input_img = cv2.resize(input_img, img_size)
input_img = np.expand_dims(input_img, axis=0)
```
- **`Image.open`**: Loads the satellite image and its corresponding mask.
- **`cv2.resize`**: Resizes the image to match the model's input size (`128x128`).
- **`np.expand_dims`**: Adds a batch dimension to the input image, preparing it for model prediction.

---

#### 3. **Predict the Mask**:
```python
predicted_mask = model.predict(input_img).reshape(img_size)
```
- Uses the trained U-Net model to predict the segmentation mask.
- **Reshaping**: Converts the model’s output into a 2D array for visualization.

---

#### 4. **Plot the Results**:
```python
plt.figure(figsize=(15, 5))

# Display original image
plt.subplot(1, 3, 1)
plt.title(f"Original Image {i+1}")
plt.imshow(image)
plt.axis("off")

# Display ground truth mask
plt.subplot(1, 3, 2)
plt.title(f"Ground Truth Mask {i+1}")
plt.imshow(mask, cmap="gray")
plt.axis("off")

# Display predicted mask
plt.subplot(1, 3, 3)
plt.title(f"Predicted Mask {i+1}")
plt.imshow(predicted_mask, cmap="gray")
plt.axis("off")
```
- Creates a subplot with three columns:
  - Original satellite image.
  - Ground truth segmentation mask.
  - Predicted segmentation mask.
- Titles and axes are added for clarity.

---

### **Why This Step is Important**:
1. **Qualitative Evaluation**:
   - Helps visually assess how well the model predicts segmentation masks compared to the ground truth.
2. **Debugging**:
   - Identifies potential issues such as misaligned predictions, incomplete masks, or over/under-segmentation.
3. **Model Improvement**:
   - Insights from these visualizations can guide further model fine-tuning or adjustments.

---

### **Outcome**:
After running this function:
1. Side-by-side comparisons of images, ground truth masks, and predictions are displayed.
2. These visualizations provide a qualitative understanding of the model’s segmentation performance.

## Evaluating the Model Using IoU (Intersection over Union)

---

### **Code**:
```python
from sklearn.metrics import jaccard_score

# Assuming y_test has continuous values (e.g., grayscale),
# threshold it to binary as well
y_test_binary = (y_test > 0.5).astype(int)

# Flatten masks for IoU calculation
iou = jaccard_score(y_test_binary.flatten(), (model.predict(X_test) > 0.5).astype(int).flatten())
print("IoU Score:", iou)
```

---

### **Purpose**:
This step calculates the **Intersection over Union (IoU)**, also known as the **Jaccard Index**, to evaluate the segmentation model's performance. IoU measures the overlap between the predicted and ground truth masks, providing a robust metric for semantic segmentation tasks.

---

### **Explanation**:

#### 1. **Threshold Ground Truth Masks**:
```python
y_test_binary = (y_test > 0.5).astype(int)
```
- **Purpose**: Converts the continuous grayscale mask values (`0-1`) in `y_test` to binary values (`0` or `1`).
  - Pixels with values > 0.5 are set to 1 (target class, e.g., water).
  - Pixels with values ≤ 0.5 are set to 0 (background).
- **Why?** IoU requires binary masks to compare the predicted and ground truth labels.

---

#### 2. **Predict Masks**:
```python
model.predict(X_test)
```
- **Purpose**: Generates predicted masks for the testing dataset (`X_test`).
- The predicted values are probabilities in the range `[0, 1]`.

---

#### 3. **Threshold Predicted Masks**:
```python
(model.predict(X_test) > 0.5).astype(int)
```
- Converts the predicted probabilities to binary values using a threshold of 0.5.
  - Values > 0.5 are set to 1.
  - Values ≤ 0.5 are set to 0.

---

#### 4. **Flatten the Masks**:
```python
y_test_binary.flatten()
(model.predict(X_test) > 0.5).astype(int).flatten()
```
- Flattens the 2D masks into 1D arrays for comparison.
- Required by `jaccard_score`, which computes the IoU for binary classification.

---

#### 5. **Calculate IoU**:
```python
iou = jaccard_score(y_test_binary.flatten(), (model.predict(X_test) > 0.5).astype(int).flatten())
```
- **Jaccard Score (IoU)**:
  - Formula:
    \[
    \text{IoU} = \frac{\text{Intersection of True Positive Pixels}}{\text{Union of True and Predicted Pixels}}
    \]
  - Measures the degree of overlap between predicted and ground truth masks.
  - A value close to 1 indicates high overlap, while a value close to 0 indicates poor overlap.

---

#### 6. **Print IoU Score**:
```python
print("IoU Score:", iou)
```
- Displays the computed IoU value, giving a quantitative measure of the model's segmentation performance.

---

### **Why This Step is Important**:
1. **Robust Evaluation**:
   - IoU is a widely used metric for segmentation tasks, providing a clear measure of overlap between predicted and true masks.

2. **Identifies Model Strength**:
   - High IoU indicates accurate segmentation, while low IoU suggests room for improvement in model architecture or training.

3. **Threshold and Flattening**:
   - Ensures compatibility between predicted and ground truth masks for a fair comparison.

---

### **Outcome**:
After running this code:
1. The IoU score for the testing dataset is calculated and printed.
2. This score quantitatively evaluates the model's segmentation accuracy.
