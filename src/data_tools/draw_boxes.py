import cv2
import numpy as np
from config import CLASSES

# Create different colors for each class.
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def draw_boxes(image, bboxes, format='voc'):
    """
    Function accepts an image and bboxes list and returns
    the image with bounding boxes drawn on it.

    Parameters
    :param image: Image, type NumPy array.
    :param bboxes: Bounding box in Python list format.
    :param format: One of 'coco', 'voc', 'yolo' depending on which final
        bounding noxes are formated.

    Return
    image: Image with bounding boxes drawn on it.
    box_areas: list containing the areas of bounding boxes.
    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    box_areas = []

    if format == 'coco':
        # coco has bboxes in xmin, ymin, width, height format
        # we need to add xmin and width to get xmax and...
        # ... ymin and height to get ymax
        for box_num, box in enumerate(bboxes):
            xmin = int(box[0])
            ymin = int(box[1])
            xmax = int(box[0])+int(box[2])
            ymax = int(box[1])+int(box[3])
            width = int(box[2])
            height = int(box[3])
            cv2.rectangle(
                image, 
                (xmin, ymin), (xmax, ymax),
                color=(0, 0, 255),
                thickness=2
            )
            box_areas.append(width*height)

    if format == 'voc':
        for box_num, box in enumerate(bboxes):
            xmin = int(box[0])
            ymin = int(box[1])
            xmax = int(box[2])
            ymax = int(box[3])
            width = xmax - xmin
            height = ymax - ymin
            cv2.rectangle(
                image, 
                (xmin, ymin), (xmax, ymax),
                color=(0, 0, 255),
                thickness=2
            )
            box_areas.append(width*height) 

    if format == 'yolo':
        # need the image height and width to denormalize...
        # ... the bounding box coordinates
        h, w, _ = image.shape
        for box_num, box in enumerate(bboxes):
            x1, y1, x2, y2 = yolo2bbox(box)
            # denormalize the coordinates
            xmin = int(x1*w)
            ymin = int(y1*h)
            xmax = int(x2*w)
            ymax = int(y2*h)
            width = xmax - xmin
            height = ymax - ymin
            cv2.rectangle(
                image, 
                (xmin, ymin), (xmax, ymax),
                color=(0, 0, 255),
                thickness=2
            ) 
            box_areas.append(width*height) 
    return image, box_areas

def yolo2bbox(bboxes):
    """
    Function to convert bounding boxes in YOLO format to 
    xmin, ymin, xmax, ymax.
    
    Parmaeters:
    :param bboxes: Normalized [x_center, y_center, width, height] list

    return: Normalized xmin, ymin, xmax, ymax
    """
    xmin, ymin = bboxes[0]-bboxes[2]/2, bboxes[1]-bboxes[3]/2
    xmax, ymax = bboxes[0]+bboxes[2]/2, bboxes[1]+bboxes[3]/2
    return xmin, ymin, xmax, ymax

def draw_boxes_video(boxes, classes, labels, image):
    """
    Draws the bounding box around a detected object.
    """
    lw = max(round(sum(image.shape) / 2 * 0.003), 2)  # Line width.
    tf = max(lw - 1, 1) # Font thickness.

    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        cv2.rectangle(
            img=image,
            pt1=(int(box[0]), int(box[1])),
            pt2=(int(box[2]), int(box[3])),
            color=color[::-1], 
            thickness=lw
        )
        cv2.putText(
            img=image, 
            text=classes[i], 
            org=(int(box[0]), int(box[1]-5)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=lw / 3, 
            color=color[::-1], 
            thickness=tf, 
            lineType=cv2.LINE_AA
        )
    return image