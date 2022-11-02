import torch
import torchvision
import cv2
import numpy as np
import os
import glob as glob
import random

from xml.etree import ElementTree as et
from config import CLASSES, RESIZE_TO, TRAIN_DIR, VALID_DIR, BATCH_SIZE, AUGMENTATION
from torch.utils.data import Dataset, DataLoader
from custom_utils import collate_fn
from data_tools.augmentations import CutOut, DropFlip, BrigTransScale
from data_tools.resize_utilities import usual_resize_image, custom_resize_image, zerop_resize_image


# the dataset class
class CustomDataset(Dataset):

    def __init__(self, images_path,width, height, classes, augmented=True):
        self.augmented = augmented
        self.images_path = images_path
        self.height = height
        self.width = width
        self.classes = classes
        self.image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm']
        self.all_image_paths = []

        # get all the image paths in sorted order
        for file_type in self.image_file_types:
            self.all_image_paths.extend(glob.glob(f"{self.images_path}/{file_type}"))
        self.all_annot_paths = glob.glob(f"{self.images_path}/*.xml")
        # Remove all annotations and images when no object is present.
        self.read_and_clean()
        self.all_images = [image_path.split(os.path.sep)[-1] for image_path in self.all_image_paths]
        self.all_images = sorted(self.all_images)

       
    def read_and_clean(self):
        """
        This function will discard any images and labels when the XML 
        file does not contain any object.
        """
        for annot_path in self.all_annot_paths:
            tree = et.parse(annot_path)
            root = tree.getroot()
            object_present = False
            for member in root.findall('object'):
                object_present = True
            if object_present == False:
                print(f"Removing {annot_path} and corresponding image")
                self.all_annot_paths.remove(annot_path)
                self.all_image_paths.remove(annot_path.split('.xml')[0]+'.jpg')
 
    

    def __getitem__(self, idx):
        
        # capture the image name and the full image path
        image_name = self.all_images[idx]
        #print(image_name)
        image_path = os.path.join(self.images_path, image_name)
        
        # read the image
        image = cv2.imread(image_path)
        # convert BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        #image_resized = cv2.resize(image, (self.width, self.height))
        #image_resized /= 255.0
        
        # capture the corresponding XML file for getting the annotations
        annot_filename = image_name[:-4] + '.xml'
        annot_file_path = os.path.join(self.images_path, annot_filename)
        
        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()

        # TEMPORAL -Just for bad annottations-
        object = False

        # get the height and width of the image
        image_width = image.shape[1]
        image_height = image.shape[0]

        #image_resized, boxes, labels = custom_resize_image(image, self.width, self.height, root)  
        image_resized, boxes, labels = zerop_resize_image(image, self.width, self.height, root)  
        #image_resized, boxes, labels = usual_resize_image(image, root, self.width, self.height)  
        image_resized /= 255.0

        # box coordinates for xml files are extracted and corrected for image size given
        """ for member in root.findall('object'):
            # map the current object name to `classes` list to get...
            # ... the label index and append to `labels` list
            labels.append(self.classes.index(member.find('name').text))
            
            # xmin = left corner x-coordinates
            xmin = int(member.find('bndbox').find('xmin').text)
            # xmax = right corner x-coordinates
            xmax = int(member.find('bndbox').find('xmax').text)
            # ymin = left corner y-coordinates
            ymin = int(member.find('bndbox').find('ymin').text)
            # ymax = right corner y-coordinates
            ymax = int(member.find('bndbox').find('ymax').text)
            
            # resize the bounding boxes according to the...
            # ... desired `width`, `height`
            xmin_final = (xmin/image_width)*self.width
            xmax_final = (xmax/image_width)*self.width
            ymin_final = (ymin/image_height)*self.height
            yamx_final = (ymax/image_height)*self.height 
            
            boxes.append([xmin_final, ymin_final, xmax_final, yamx_final]) """
    

        
        # bounding box to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # area of the bounding boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        #  no crowd instances
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        # labels to tensor
        labels = torch.as_tensor(labels, dtype=torch.int64)
        

        # prepare the final `target` dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id


        # apply the image transforms or transform image as Tensor
        if self.augmented is not False:

            print("DATA AUGMENTATION")
            #print(target['boxes'])


            if random.random() < AUGMENTATION:
                n = random.randint(0, 2)

                if n == 0:
                    image_resized, bboxes = CutOut()(image_resized, target["boxes"])
                elif n == 1:
                    image_resized, bboxes = DropFlip()(image_resized, target["boxes"])
                else:
                    image_resized, bboxes = BrigTransScale()(image_resized, target["boxes"])
                
                target['boxes'] = torch.as_tensor(bboxes, dtype=torch.float32)

                #print("AFTER DATA AUGMENTATION")
                #print(target['boxes'])

            image_resized = torchvision.transforms.ToTensor()(image_resized.copy())
            
            
        else:
            image_resized = torchvision.transforms.ToTensor()(image_resized)

            
        return image_resized, target


    def __len__(self):
        return len(self.all_images)




# prepare the final datasets and data loaders
def create_train_dataset():
    train_dataset = CustomDataset(TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES,True)
    #train_dataset = MicrocontrollerDataset(TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_train_transform())
    return train_dataset

def create_valid_dataset():
    valid_dataset = CustomDataset(VALID_DIR, RESIZE_TO, RESIZE_TO, CLASSES, False)
    #valid_dataset = MicrocontrollerDataset(VALID_DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_valid_transform())
    return valid_dataset

def create_train_loader(train_dataset, num_workers=0):
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    return train_loader

def create_valid_loader(valid_dataset, num_workers=0):
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
        )
    return valid_loader


# execute datasets.py using Python command from Terminal...
# ... to visualize sample images
# USAGE: python datasets.py
if __name__ == '__main__':
    # sanity check of the Dataset pipeline with sample visualization
    dataset = CustomDataset(
        TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES
    )
    print(f"Number of training images: {len(dataset)}")
    
    # function to visualize a single sample
    def visualize_sample(image, target):
        for box_num in range(len(target['boxes'])):
            box = target['boxes'][box_num]
            label = CLASSES[target['labels'][box_num]]
            cv2.rectangle(
                image, 
                (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                (0, 255, 0), 2
            )
            cv2.putText(
                image, label, (int(box[0]), int(box[1]-5)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        
    NUM_SAMPLES_TO_VISUALIZE = 5
    for i in range(NUM_SAMPLES_TO_VISUALIZE):
        image, target = dataset[i]
        visualize_sample(image, target)