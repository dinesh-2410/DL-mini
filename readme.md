# README.md for Object Detection Project

## Overview

This repository contains a Python-based object detection project utilizing the Faster R-CNN model from the PyTorch library. The project implements image processing techniques to detect objects in various images and draw bounding boxes around them. 

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Functions](#functions)
- [Model Details](#model-details)
- [Images Used](#images-used)
- [Results](#results)
- [License](#license)

## Installation

To set up the environment for this project, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/dinesh-2410/DL-mini.git
   cd DL-mini
   ```

2. **Install required packages:**

   ```bash
   ! pip install -q condacolab
   import condacolab
   condacolab.install()
   ! conda install pytorch=1.1.0 torchvision -c pytorch -y
   ```
3. **Install additional dependencies:**

   ```bash
   pip install opencv-python numpy pillow matplotlib
   ```
## Usage

1. **Download images:**

   Use the following commands to download sample images:
   ```bash
   !wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-Coursera/images%20/images_part_5/DLguys.jpeg
   !wget https://www.ajot.com/images/uploads/article/quantas-car-v-plane-TAKE-OFF.jpg
   !wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-Coursera/images%20/images_part_5/istockphoto-187786732-612x612.jpeg
   !wget https://cdn.webrazzi.com/uploads/2015/03/andrew-ng1.jpg
   ```

2. **Run the object detection script:**

   Execute the main script to perform object detection on the downloaded images.
   ```bash
   python main.py
   ```

## Functions
   `get_predictions(pred, threshold=0.8, objects=None)`

   * Filters predictions based on a confidence threshold and specified object classes.

   `draw_box(pred_class, img, rect_th=2, text_size=0.5, text_th=2, download_image=False, img_name="img")`

   * Draws bounding boxes around detected objects in the image.

   `save_RAM(image_=False)`

   * Releases memory used by images and predictions.

## Model Details
The project utilizes the Faster R-CNN model with a ResNet-50 backbone pre-trained on the COCO dataset. The model is capable of detecting various objects, including but not limited to:

   * Person
   * Bicycle
   * Car
   * Dog
   * Cat

The COCO_INSTANCE_CATEGORY_NAMES array contains the list of object classes the model can detect.

## Images Used
The following images are used for object detection in this project:

   * Andrew Ng Image: andrew-ng1.jpg
   * DL Guys Image: DLguys.jpeg
   * Istockphoto Image: istockphoto-187786732-612x612.jpeg
   * Quantas Car vs Plane Image: quantas-car-v-plane-TAKE-OFF.jpg

## Results
The results include bounding boxes drawn around detected objects in each image, with the class label and detection probability displayed. The processed images are saved with bounding boxes for further analysis.

### Example Results
   * Andrew Ng Image: Detected "person" with high confidence.
   * DL Guys Image: Detected "person" with high confidence.
   * Istockphoto Image: Detected "dog", "cat", and "bird".
   * Quantas Car vs Plane Image: Detected "car" and "airplane".

## License
This project is licensed under the MIT License - see the LICENSE file for details.