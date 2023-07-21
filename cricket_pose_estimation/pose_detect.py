# imports library

import argparse
import cv2
import glob
from os.path import splitext
import os
from ultralytics import YOLO
import numpy as np
import pandas as pd
import joblib

'''
Parameters that are required to run this program
'''
parser = argparse.ArgumentParser()
parser.add_argument('--img_dir', type=str, default='./', help='give your directory name')
parser.add_argument('--weights', type=str, default='yolov8n-pose.pt', help='')
args = parser.parse_args()


def check_data_format(directory):
    # in our case we considering only image files

    has_video = False
    has_image = False

    for file in os.listdir(directory):
        filename, ext = splitext(file)
        
        if ext in ['.mp4', '.mov']:
            has_video = True
            
        if ext in ['.jpg', '.jpeg', '.png']:
            has_image = True
            
    if (has_video==False) and (has_image == False):
        raise ValueError('Directory should contain either video or image data')
    elif (has_video==False) and has_image:
        return True
    return False

def save_img(text="put text",img_path="./"):
    # Configure font and text
    font = cv2.FONT_HERSHEY_SIMPLEX 
    org = (50, 50) 
    fontScale = 1
    color = (255, 0, 0) 
    thickness = 2

    img_name = img_path.split("/")[-1]
    print(f"img name = {img_name}")

    # Write text on image
    image = cv2.imread(img_path)
    # image = cv2.putText(image, text, org, font, fontScale, color, thickness, cv2.LINE_AA)

    height, width, _ = image.shape
    text_x = min(50, width - 20) 
    text_y = min(50, height - 20)
    (text_width, text_height), _ = cv2.getTextSize(text, font, fontScale, thickness)
    text_x = max(50, width - text_width - 20)  
    text_y = max(50, height - text_height - 20)
    # bg_color = image[text_y, text_x]

    cv2.putText(image, text, (text_x, text_y), font, fontScale, color, thickness)

    # Get current working directory 
    cwd = os.getcwd()

    # Save image to directory
    save_path = os.path.join(cwd, 'output')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_image_path = os.path.join(save_path, img_name) 
    print(f"save img path = {save_image_path}")
    cv2.imwrite(save_image_path, image) 

    print('Image saved successfully to:', save_image_path)

    

'''
Reading all the images in a directory and stores in a image_files list
'''
image_extensions = ['.jpg', '.jpeg', '.png']
image_files = []

if check_data_format(args.img_dir) == True:
    for file in os.listdir(args.img_dir):
        filename, ext = splitext(file)
        if ext.lower() in image_extensions:
            image_files.append(os.path.join(args.img_dir,file))


# Loading YOLO-pose model for pose estimation and keypoints
model_pose = YOLO(args.weights)

# keypoint detection 
for i in image_files:
    i = i.replace("\\","/")
    results = model_pose(i,conf=0.50,save=False)
    keypoints = results[0].keypoints
    tensor_data = np.array(keypoints.xy)
        
    flattened_array = tensor_data.flatten()
    flattened_array = flattened_array.reshape(1,-1)
    df = pd.DataFrame(flattened_array)


    # load pose_model.pkl
    rf_classifier = joblib.load('./model/pose_model1.pkl')

    value = int(rf_classifier.predict(df))
    
    dict ={0:"coverdrive",
           1:"sweep"}


    with open("output.txt", "a") as file:
        # file.write("flatened= " + str(flattened_array) + "\n")
        # file.write("df= " + str(df) + "\n")
        file.write("prediction= "+str(args.img_dir) + str(rf_classifier.predict(df)) + "\n")

    save_img(dict[value],i)
    

