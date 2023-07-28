from ultralytics import YOLO
import cv2
import os
import pandas as pd
import numpy as np

model = YOLO('yolov8n-pose.pt')

directory = "D:/Cricket_Animation_project/abdul/cric_shot/batsman_SWEEp"
class_name = "sweep"

# results = model('D:/Cricket_Animation_project/yolov5/testing/cr2.mp4',stream=False,conf=0.50,save=True,save_crop=True)
# df = {}
for filename in os.listdir(directory):
    results = model(f'{directory}/{filename}',conf=0.50,save=False)
    keypoints = results[0].keypoints
    tensor_data = np.array(keypoints.xy)
    
    flattened_array = tensor_data.flatten()
    # flattened_array = flattened_array.reshape(1,-1)

    # print(flattened_array)
    array_with_text = np.concatenate((flattened_array.astype(str), np.array([class_name])), axis=0)
    array_with_text = np.concatenate((array_with_text.astype(str), np.array([filename])), axis=0)

    array_with_text = [array_with_text]

    df = pd.DataFrame(array_with_text)  # Convert tensor data to a pandas DataFrame
   
    # with open("outputSingle.txt", "a") as file:
    # #     # file.write("keypoints xy= " + str(keypoints.xy) + "/n")
    # #     file.write(f"tensor data = {tensor_data} /n/n")
    #     file.write(f"flatened array = {flattened_array}/n/n/n")
    #     file.write(f"arraywithtext = {array_with_text}/n")
    #     file.write(f"df = {df}")

    df.to_csv('keypoints.csv', mode='a', header = False, index=False)

# reading csv file 
# Read the CSV file (if the original file doesn't have a header, set header=None)
df = pd.read_csv("keypoints.csv", header=None)

# # Add header to the DataFrame
header_list = ["key_" + str(i) for i in range(1, len(df.columns) + 1)]
df.columns = header_list

# # Save the DataFrame to a new CSV file with the header
df.to_csv("keypoints.csv", index=False)

print("CSV file with a header has been saved to keypoints.csv")


# keypoints = results[0].keypoints  # Masks object
# keypoints.xy  # x, y keypoints (pixels), (num_dets, num_kpts, 2/3), the last dimension can be 2 or 3, depends the model.
# keypoints.xyn  # x, y keypoints (normalized), (num_dets, num_kpts, 2/3)
# keypoints.conf  # confidence score(num_dets, num_kpts) of each keypoint if the last dimension is 3.
# keypoints.data  # raw keypoints tensor, (num_dets, num_kpts, 2/3) 


# res_plotted = results[0].plot()
# cv2.imshow("result", res_plotted)
