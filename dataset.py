from config import *
import numpy as np
import json
import cv2
import os

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F



class AarizDataset(Dataset):

    def __init__(self, dataset_folder_path: str, mode: str):
        
        if (mode == "TRAIN") or (mode == "VALID") or (mode == "TEST"):
            mode = mode.capitalize()
        else:
            raise ValueError("mode could only be TRAIN, VALID or TEST")
        
        
        self.images_root_path = os.path.join(dataset_folder_path, mode, "Cephalograms")
        self.labels_root_path = os.path.join(dataset_folder_path, mode, "Annotations")
        
        self.senior_annotations_root = os.path.join(self.labels_root_path, "Cephalometric Landmarks", "Senior Orthodontists")
        self.junior_annotations_root = os.path.join(self.labels_root_path, "Cephalometric Landmarks", "Junior Orthodontists")
        self.cvm_annotations_root = os.path.join(self.labels_root_path, "CVM Stages")
        
        self.images_list = os.listdir(self.images_root_path)
        self.target_size = 512
        self.h_max, self.w_max = 2750, 2200
        
    
    def __getitem__(self, index):
        image_file_name = self.images_list[index]
        # get name of image file and combine to get corresponding json file.
        label_file_name = self.images_list[index].split(".")[0] + "." + "json"
        image, (scale_x, scale_y) = self.get_image(image_file_name)
        landmarks = self.get_landmarks(label_file_name, (scale_x, scale_y))
        cvm_stage = self.get_cvm_stage(label_file_name)
        
        return image, landmarks, cvm_stage

    # def __pad_image(self, image, h_max=2300, w_max=2200):
    #
    #     pad_bottom = h_max - image.shape[1]
    #     pad_right = w_max - image.shape[2]
    #
    #     pad = (0,pad_right,0,pad_bottom)
    #
    #     padded_image = F.pad(image, pad)


        return padded_image

    def get_image(self, file_name: str):
        file_path = os.path.join(self.images_root_path, file_name)
        
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h,w,c = image.shape

        padded = np.zeros((self.h_max, self.w_max, 3), dtype=np.uint8)
        padded[:h, :w, :] = image

        # resize to 512 by 512 for ResNet

        resized = cv2.resize(padded, (self.target_size, self.target_size))

        # convert to float tensor with RGB values reduced to values between 0 and 1
        image_tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0

        # scale factor for co-ordinates to get to 512 by 512
        scale_factor = (self.target_size/self.w_max, self.target_size/self.h_max)

        # Conversion to tensor from numpy array
        # (C,H,W)
        # image = torch.from_numpy(padded).permute(2, 0, 1)  # [C,H,W]
        return image_tensor, scale_factor
    
    
    def get_landmarks(self, file_name,scale_factor):
        file_path = os.path.join(self.senior_annotations_root, file_name)
        with open(file_path, mode="r") as file:
            senior_annotations = json.load(file)
        
        senior_annotations = [[landmark["value"]["x"], landmark["value"]["y"]] for landmark in senior_annotations["landmarks"]]
        senior_annotations = np.array(senior_annotations, dtype=np.float32)
        
        file_path = os.path.join(self.junior_annotations_root, file_name)
        with open(file_path, mode="r") as file:
            junior_annotations = json.load(file)

        junior_annotations = [[landmark["value"]["x"], landmark["value"]["y"]] for landmark in junior_annotations["landmarks"]]
        junior_annotations = np.array(junior_annotations, dtype=np.float32)
        
        # landmarks = np.zeros(shape=(NUM_LANDMARKS, 2), dtype=np.float64)

        # combine and average landmark classification of juniors and seniors
        # landmarks[:, 0] = np.ceil((0.5) * (junior_annotations[:, 0] + senior_annotations[:, 0]))
        # landmarks[:, 1] = np.ceil((0.5) * (junior_annotations[:, 1] + senior_annotations[:, 1]))

        # average x co-ordinates of junior and senior for each landmark
        avg_x = (junior_annotations[:, 0] + senior_annotations[:, 0]) / 2.0
        # average y co-ordinates of junior and senior per landmark
        avg_y = (junior_annotations[:, 1] + senior_annotations[:, 1]) / 2.0

        # landmarks = np.array(landmarks, dtype=np.float32)[np.newaxis, :, :]
        # (29,2)
        landmarks = np.stack([avg_x, avg_y], axis=1)

        landmarks[:, 0] = landmarks[:, 0] * scale_factor[0]
        landmarks[:, 1] = landmarks[:, 1] * scale_factor[1]

        # dim (1, 29, 2)
        landmarks = torch.from_numpy(np.array(landmarks, dtype=np.float32))

        # reduce co-ordinates to within 0 and 1 by dividing by 512
        landmarks = landmarks / self.target_size

        # dim (29,2)
        landmarks = landmarks.squeeze(0)

        return landmarks

    
    def get_cvm_stage(self, file_name):
        file_path = os.path.join(self.cvm_annotations_root, file_name)
        
        with open(file_path, mode="r") as file:
            cvm_annotations = json.load(file)
        
        cvm_stage_value = cvm_annotations["cvm_stage"]["value"]
        cvm_stage = np.zeros(shape=(NUM_CVM_STAGES, ))
        cvm_stage[cvm_stage_value - 1] = 1.0
        
        return np.array(cvm_stage, dtype=np.float32)[np.newaxis, :]
    
    def __len__(self):
        return len(self.images_list)
