import os
import sys
import random
import math

import skimage.io

import matplotlib.pyplot as plt
%matplotlib inline

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn import config

import pandas as pd
import numpy as np

from PIL import Image
from sklearn.cluster import KMeans
from collections import Counter

# Root directory of the model
ROOT_DIR = os.path.abspath('file_path')
# Import Mask RCNN
sys.path.append(ROOT_DIR)
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# ------- Configure model
config = config.Config()
config.NAME = "coco"

config.STEPS_PER_EPOCH = 1
config.NUM_CLASSES = 81
config.BATCH_SIZE = 1
config.IMAGE_META_SIZE = 93
config.RPN_TRAIN_ANCHORS_PER_IMAGE = 100 ### can decrease it
config.IMAGES_PER_GPU = 1

# ------- Create model and load trained weights
# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# ------- Create functions
def image_mask(image):
    # detect person and obtain mask
    results = model.detect([image], verbose=1)
    r = results[0]
    mask = r["masks"][:, :, 0].astype(int)

    # multiply mask with original image to exclude the background
    image2 = np.multiply(image.swapaxes(0, 2), mask.swapaxes(0, 1))
    image2 = image2.swapaxes(0, 2)

    return image2

def extract_person(image):

    # convert image to RGBA
    img = Image.fromarray(np.uint8(image)).convert("RGBA")

    # make background pixels transparent
    pixdata = img.load()

    width, height = img.size
    for y in range(height):
        for x in range(width):
            if pixdata[x, y] == (0, 0, 0, 255):
                pixdata[x, y] = (0, 0, 0, 0)

    # change resulting image to np array
    pix = np.array(img)
    modified_image = pix.reshape(pix.shape[0] * pix.shape[1], 4)

    # delete transparent pixels
    new = pd.DataFrame(modified_image)
    good = new[new[3] != 0]  # delete transparent pixels
    good = good.drop(columns=3, axis=1)

    # return the array of colored pixels
    colors = good.to_numpy()

    return colors


def extract_colors(image):
    clf = KMeans(n_clusters=3)
    labels = clf.fit_predict(image)

    counts = Counter(labels)

    # get top colors
    center_colors = clf.cluster_centers_
    list1 = center_colors.tolist()

    # save top colors as list, round decimals
    colors_final = []
    for l in list1:
        for e in l:
            colors_final.append(round(e))

    # create final list with colors
    list_to_put_in_frame = colors_final

    return list_to_put_in_frame

# ------- Read image folder
IMAGE_DIR = os.path.abspath('file_path')
# Extract all the file names from the path
list_files = []
list_files=os.listdir(IMAGE_DIR)
list_files.sort()
list_files=list_files[1:] # to delete the DS.store file

len(list_files)

# Run
color_list = []

for i in range(0, len(list_files)):
    image_name = list_files[i]
    image_path = os.path.join(IMAGE_DIR, image_name)
    image = plt.imread(image_path)

    new_image = image_mask(image)
    person = extract_person(new_image)
    colors = extract_colors(person)

    color_list.append(colors)

# Save
color_frame=pd.DataFrame(color_list, columns=["R1","G1","B1","R2","G2","B2","R3","G3","B3"])
color_frame["image_name"]=list_files

color_frame.to_csv("name.csv",index=False)

