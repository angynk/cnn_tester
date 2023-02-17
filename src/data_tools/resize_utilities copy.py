import os
import cv2
import numpy as np
import random
from data_tools.bbox_composer import get_bbox_usual_resize, get_bbox, get_bbox_custom_compose



# USUAL RESIZE USING CV2 AND INTERPOLATION
def usual_resize_image(img_original, root, target_width, target_height):
    height,width,_ = img_original.shape
    target_dim = (target_width, target_height)
    resized_image = cv2.resize(img_original, target_dim, interpolation = cv2.INTER_AREA)
    boxes, labels = get_bbox_usual_resize(root, target_width, target_height, width, height) 
    return resized_image, boxes, labels   

# CUSTOM RESIZE USING OTHER IMAGES

def random_point(max_value):
    value = random.randint(0,max_value)
    return value

def get_img_helper():
    value = random_point(9)
    img_helper = cv2.imread(f"dataset/helpers/road_{value}.jpg")   
    return img_helper

def crop (width, height, image):
    h2, w2 = image.shape[:2]
    ini_h = random_point(h2-height)
    ini_w = random_point(w2-width)
    crop_image = image[ini_h:height+ini_h, ini_w:width+ini_w]
    return crop_image

def get_shape_img(image_base, image_add):
    original_first = bool(random.getrandbits(1))
    if original_first:
            h1, w1 = image_base.shape[:2]
            h2, w2 = image_add.shape[:2]
            first_image = image_base
            second_image = image_add
    else:
        h1, w1 = image_add.shape[:2]
        h2, w2 = image_base.shape[:2]
        first_image = image_add
        second_image = image_base  
    
    return h1,w1,h2,w2,first_image,second_image, original_first

def merge_images(helper_backgroud, image_base, target_width, target_height, root):
    resized_helper = cv2.resize(helper_backgroud, (target_width, target_height), interpolation = cv2.INTER_AREA)
    resized_helper = resized_helper.astype(np.float32)
    h2, w2 = image_base.shape[:2]
    ini_h = random_point(target_height-h2)
    ini_w = random_point(target_width-w2)
    resized_helper[ini_h:h2+ini_h,ini_w:w2+ini_w,:] = image_base[0:h2,0:w2,:]
    boxes, labels = get_bbox_custom_compose(root,ini_w ,ini_h)
    return resized_helper, boxes, labels

def composeh_image (image_base,image_add, root, extra_height):

    h1,w1,h2,w2,first_image,second_image, original_first = get_shape_img(image_base,image_add)
    resize_image = np.zeros((h1+h2,max(w1,w2),  3), np.uint8)
    resize_image[:h1, :w1,:3] = first_image
    resize_image[h1:h1+h2, :w2,:3] = second_image
    resize_image = resize_image.astype(np.float32)

    if original_first is False:
        boxes, labels = get_bbox_custom_compose(root, 0, extra_height)
    else: 
        boxes, labels = get_bbox(root)

    return resize_image, boxes, labels

def composev_image (image_base,image_add, root, extra_width):

    h1,w1,h2,w2,first_image,second_image, original_first = get_shape_img(image_base,image_add)
    resize_image = np.zeros((max(h1,h2), w1+w2,  3), np.uint8)
    resize_image[:h1, :w1,:3] = first_image
    resize_image[:h2, w1:w1+w2,:3] = second_image
    resize_image = resize_image.astype(np.float32)

    if original_first is False:
        boxes, labels = get_bbox_custom_compose(root,extra_width, 0)
    else: 
        boxes, labels = get_bbox(root)

    return resize_image, boxes, labels


def custom_resize_image(img_original, target_width,target_height, root):
    img_helper = get_img_helper()
    height, width, _ = img_original.shape 
    print("Size: "+str(width)+" x "+str(height))
    resized_image = img_original

    if width > target_width:
        resized_image, boxes, labels = usual_resize_image(img_original, root, target_width, target_height)
    elif width < target_width:
        if height > target_height:
            resized_image, boxes, labels = usual_resize_image(img_original, root, target_width, target_height)
        elif height < target_height:
            resized_image, boxes, labels = merge_images(img_helper, img_original, target_width, target_height, root)
        else:
            extra_width = target_width - width
            crop_image = crop (extra_width, target_height, img_helper)
            resized_image, boxes, labels = composev_image(img_original,crop_image,root, extra_width)
    else:
        if height > target_height:
            resized_image, boxes, labels = usual_resize_image(img_original, root, target_width, target_height)
        elif height < target_height:
            extra_heigth = target_height - height
            crop_image = crop (target_width, extra_heigth, img_helper) 
            resized_image, boxes, labels = composeh_image(img_original,crop_image,root, extra_heigth)

    return resized_image, boxes, labels


# RESIZE WITH ZERO PADDING


def complete_horizontal(size):
    complete_left = bool(random.getrandbits(1))
    left = 0
    rigth = 0
    if complete_left:
        left = size
    else:
        rigth = size
    
    return left,rigth

def complete_vertical(size):
    complete_top = bool(random.getrandbits(1))
    top = 0
    bottom = 0
    if complete_top:
        top = size
    else:
        bottom = size
    
    return top,bottom

def complete_random_border(width, height , target_width, target_height):
    left = random_point(target_width - width)
    rigth = target_width - left
    top = random_point(target_height -height)
    bottom = target_height - top

    return top, bottom, left, rigth

def zerop_resize_image(img_original, target_width,target_height, root):
    height, width, _ = img_original.shape 
    print("Size: "+str(width)+" x "+str(height))
    resized_image = img_original
    color = [0, 0, 0]

    if width > target_width:
        resized_image, boxes, labels = usual_resize_image(img_original, root, target_width, target_height)
    elif width < target_width:
        if height > target_height:
            resized_image, boxes, labels = usual_resize_image(img_original, root, target_width, target_height)
        elif height < target_height:
            top, bottom, left, right = complete_random_border(width, height, target_width, target_height)
            resized_image = cv2.copyMakeBorder(img_original, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)
            boxes, labels = get_bbox_custom_compose(root,left, top)
        else:
            extra_width = target_width - width
            left, right = complete_horizontal(extra_width)
            resized_image = cv2.copyMakeBorder(img_original, 0, 0, left, right, cv2.BORDER_CONSTANT,value=color)
            print(left)
            print(right)
            boxes, labels = get_bbox_custom_compose(root,left, 0)
    else:
        if height > target_height:
            resized_image, boxes, labels = usual_resize_image(img_original, root, target_width, target_height)
        elif height < target_height:
            extra_heigth = target_height - height
            
            top, bottom = complete_vertical(extra_heigth)
            print(top)
            print(bottom)
            resized_image = cv2.copyMakeBorder(img_original, top, 0, bottom, 0, cv2.BORDER_CONSTANT,value=color)
            boxes, labels = get_bbox_custom_compose(root,0, top)
            

    return resized_image, boxes, labels  
