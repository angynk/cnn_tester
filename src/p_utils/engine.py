import math
from ssl import cert_time_to_seconds
import sys
import time

import torch

from config import CLASSES
from custom_utils import plot_pr_curve

from p_utils import utils
from pprint import pprint

from p_utils.coco_eval import CocoEvaluator
from p_utils.coco_utils import get_coco_api_from_dataset, _get_iou_types, summarize
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import numpy as np


def train_one_epoch(
    model, 
    optimizer, 
    data_loader, 
    device, 
    epoch, 
    train_loss_hist,
    print_freq, 
    scaler=None,
    scheduler=None
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    
    # List to store batch losses.
    batch_loss_list = []

    
    step_counter = 0

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()

        losses.backward()
        optimizer.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        batch_loss_list.append(loss_value)
        train_loss_hist.send(loss_value)


    return metric_logger, batch_loss_list


@torch.inference_mode()
def evaluate(model, data_loader, device,classes_name):

    map_metric = MeanAveragePrecision()
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

     # List to store batch losses.
    val_loss_list = []

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
        map_metric.update(outputs, targets)
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    evaluate_model(map_metric, coco_evaluator, n_threads)
    

    return coco_evaluator, val_loss_list


def evaluate_model (map_metric, coco_evaluator, n_threads):
    map = map_metric.compute()
    print("MAP: "+str(map['map'].item()))
    print("MAP 0.50: "+str(map['map_50'].item()))
    print("MAP 0.75: "+str(map['map_75'].item()))
    #pprint(map)

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    # Summarize
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)

    coco_eval = coco_evaluator.coco_eval["bbox"]
    coco_stats, print_coco = summarize(coco_eval)

    # calculate voc info for every classes(IoU=0.5)
    voc_map_info_list = []
    for i in range(len(CLASSES)):
        stats, _ = summarize(coco_eval, catId=i)
        voc_map_info_list.append(" {:15}: {}".format(CLASSES[i], stats[1]))

    print_voc = "\n".join(voc_map_info_list)
    print("Average MAP by Class IoU=0.5")
    print(print_voc)

    # Plot Precision/Recall Curve
    plot_pr_curve(coco_eval, CLASSES)


def predict(image, model, device, detection_threshold):
    """
    Predict the output of an image after forward pass through
    the model and return the bounding boxes, class names, and 
    class labels. 
    """
    # Transform the image to tensor.
    image =image.to(device)
    # Add a batch dimension.
    image = image.unsqueeze(0) 
    # Get the predictions on the image.
    with torch.no_grad():
        outputs = model(image) 

    # Get score for all the predicted objects.
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()

    # Get all the predicted bounding boxes.
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    # Get boxes above the threshold score.
    boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)
    labels = outputs[0]['labels'][:len(boxes)]
    # Get all the predicited class names.
    pred_classes = [CLASSES[i] for i in labels.cpu().numpy()]

    return boxes, pred_classes, labels