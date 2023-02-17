from xml.etree import ElementTree as et
from config import CLASSES


def get_values (member):
    # xmin = left corner x-coordinates
    xmin = int(member.find('bndbox').find('xmin').text)
    # xmax = right corner x-coordinates
    xmax = int(member.find('bndbox').find('xmax').text)
    # ymin = left corner y-coordinates
    ymin = int(member.find('bndbox').find('ymin').text)
    # ymax = right corner y-coordinates
    ymax = int(member.find('bndbox').find('ymax').text)
    return xmin, xmax, ymin, ymax

def get_bbox (root):
    boxes = []
    labels = []
    for member in root.findall('object'):
        labels.append(CLASSES.index(member.find('name').text))
        xmin, xmax, ymin, ymax = get_values(member)
        boxes.append([xmin, ymin, xmax, ymax])
    return boxes, labels


def get_bbox_usual_resize(root, target_width, target_height, width, height):
    
    boxes = []
    labels = []

    # box coordinates for xml files are extracted and corrected for image size given
    for member in root.findall('object'):

        labels.append(CLASSES.index(member.find('name').text))
            
        xmin, xmax, ymin, ymax = get_values(member)
            
        # resize the bounding boxes according to the...
        # ... desired `width`, `height`
        xmin_final = (xmin/width)*target_width
        xmax_final = (xmax/width)*target_width
        ymin_final = (ymin/height)*target_height
        yamx_final = (ymax/height)*target_height
            
        boxes.append([xmin_final, ymin_final, xmax_final, yamx_final])
    
    return boxes, labels


def get_bbox_custom_compose(root, extra_width, extra_height):
    
    boxes = []
    labels = []

    # box coordinates for xml files are extracted and corrected for image size given
    for member in root.findall('object'):

        labels.append(CLASSES.index(member.find('name').text))
            
        xmin, xmax, ymin, ymax = get_values(member)
            
        xmin_final = xmin + extra_width
        xmax_final = xmax + extra_width
        ymin_final = ymin + extra_height
        ymax_final = ymax + extra_height
            
        boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])
    
    return boxes, labels
