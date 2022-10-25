import torchvision.transforms as transforms
import albumentations as A
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

from albumentations.pytorch import ToTensorV2
from config import DEVICE, CLASSES as classes


plt.style.use('ggplot')

# This class keeps track of the training and validation loss values ...
# Helps to get the average for each epoch as well
class Averager:

    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0
    
    def send(self,value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

# To handle the data loading as different images may have different number of objects and to handle varying size tensors as well
# This helps during data loading
# handle the images and bounding boxes of varying sizes
def collate_fn(batch):
    # zip -> returns an iterator
    # tuple -> create a tuple - ordered and inmutable sequence type
    return tuple(zip(*batch))

    
# Define the training transforms
# Augmentations
def get_train_transform():
    return transforms.Compose([
        A.HorizontalFlip(0.5),
        #A.RandomRotate90(0.5),
        #A.MotionBlur(p=0.2),
        #A.MedianBlur(blur_limit=3, p=0.1),
        #A.Blur(blur_limit=3, p=0.1),
        A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ToTensorV2(p=1.0)
        
    ], bbox_params ={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })

# Define the validation transforms
def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params = {
        'format': 'pascal_voc',
        'label_fields': ['labels']
    }
    )


# This function shows the transformed images from the `train_loader`
# Helps to check whether the tranformed images along with the corresponding labels are correct or not.
def show_tranformed_image(train_loader):

    if len(train_loader) > 0:
        for i in range(1):
            images, targets = next(iter(train_loader))
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            boxes = targets[i]['boxes'].cpu().numpy().astype(np.int32)
            labels = targets[i]['labels'].cpu().numpy().astype(np.int32)
            sample = images[i].permute(1, 2, 0).cpu().numpy()
            sample = cv2.cvtColor(sample, cv2.COLOR_RGB2BGR)
        for box_num, box in enumerate(boxes):
                cv2.rectangle(sample,
                            (box[0], box[1]),
                            (box[2], box[3]),
                            (0, 0, 255), 2)
                cv2.putText(sample, CLASSES[labels[box_num]], 
                            (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 
                            1.0, (0, 0, 255), 2)
        cv2.imshow(
                'Transformed image', 
                sample
        )
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class SaveBestModel:
    
    def __init__( self, best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss
        
    def __call__(self, current_valid_loss,epoch, model, optimizer):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save(model.state_dict(), 'outputs/best_model.pth')
            

def save_model(epoch, model, optimizer):
    """
    Function to save the trained model till current epoch, or whenver called
    """
    torch.save(model.state_dict(), 'outputs/last_model.pth')


def save_loss_plot(OUT_DIR, train_loss_list):
    figure_1, train_ax = plt.subplots()
    train_ax.plot(train_loss_list, color='tab:blue')
    train_ax.set_xlabel('iterations')
    train_ax.set_ylabel('train loss')
    figure_1.savefig(f"{OUT_DIR}/train_loss.png")
    print('SAVING PLOTS COMPLETE...')
    plt.close('all')

def save_plots(train_acc, valid_acc, train_loss, valid_loss, name=None):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # Accuracy plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='tab:blue', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='tab:red', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join('outputs', name+'_accuracy.png'))
    
    # Loss plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='tab:blue', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='tab:red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join('outputs', name+'_loss.png'))



def draw_boxes(image, box, color, resize=None):
    """
    This function will annotate images with bounding boxes 
    based on wether resizing was applied to the image or not.

    :param image: Image to annotate.
    :param box: Bounding boxes list.
    :param color: Color to apply to the bounding box.
    :param resize: Either None, or provide a single integer value,
                   if 300, image will be resized to 300x300 and so on.

    Returns:
           image: The annotate image.
    """
    if resize is not None:
        cv2.rectangle(image,
                    (
                        int((box[0]/resize)*image.shape[1]), 
                        int((box[1]/resize)*image.shape[0])
                    ),
                    (
                        int((box[2]/resize)*image.shape[1]), 
                        int((box[3]/resize)*image.shape[0])
                    ),
                    color, 2)
        return image
    else:
        cv2.rectangle(image,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    color, 2)
        return image


def put_class_text(image, box, class_name, color, resize=None):
    """
    Annotate the image with class name text.

    :param image: The image to annotate.
    :param box: List containing bounding box coordinates.
    :param class_name: Text to put on bounding box.
    :param color: Color to apply to the text.
    :param resize: Whether annotate according to resized coordinates or not.

    Returns:
           image: The annotated image.
    """
    if resize is not None:
        cv2.putText(image, class_name, 
                    (
                        int(box[0]/resize*image.shape[1]), 
                        int(box[1]/resize*image.shape[0]-5)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 
                    2, lineType=cv2.LINE_AA)
        return image
    else:
        cv2.putText(image, class_name, 
                    (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 
                    2, lineType=cv2.LINE_AA)
        return image

def plot_mc_curve(px, py, save_dir='mc_curve.png', names=(), xlabel='Confidence', ylabel='Metric'):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric)

    y = py.mean(0)
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(os.path.join('outputs',save_dir), dpi=250)

def plot_pr_curve(coco_eval,nclasses,save_dir='precision.png'):
    # Precision  

    precisions = coco_eval.eval["precision"]
    fig, ax = plt.subplots()
    iou = 0 # Thresholds 0.5
    x = np.arange(0.0, 1.01, 0.01)

    for i in range(len(nclasses)):
        pr_array = precisions[iou, :, i, 0, 2] 
        # plot PR curve
        ax.plot(x, pr_array, label=nclasses[i])

    ax.grid(True)
    ax.legend(loc="lower left")
    fig.savefig(os.path.join('outputs',save_dir), dpi=250)