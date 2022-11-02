from config import CLASSES, DEVICE, NUM_CLASSES, NUM_EPOCHS, OUT_DIR, NUM_WORKERS,OUT_DIR, VISUALIZE_TRANSFORMED_IMAGES
from p_utils.engine import train_one_epoch, evaluate
from datasets import create_train_dataset, create_valid_dataset, create_train_loader, create_valid_loader
from model import create_model
from custom_utils import Averager, SaveBestModel, save_model, save_loss_plot_2, show_tranformed_image
from tqdm.auto import tqdm
import torch
import datetime

import matplotlib.pyplot as plt
import time

# MAIN CODE
print("INICIO ENTRENAMIENTO")
if __name__ == '__main__':

    total_time = time.time()
    train_dataset = create_train_dataset()
    valid_dataset = create_valid_dataset()
    train_loader = create_train_loader(train_dataset, NUM_WORKERS)
    valid_loader = create_valid_loader(valid_dataset, NUM_WORKERS)
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(valid_dataset)}\n")

    if VISUALIZE_TRANSFORMED_IMAGES:
        show_tranformed_image(train_loader)

    # Initialize the Averager class.
    train_loss_hist = Averager()
    # Train and validation loss lists to store loss values of all
    # iterations till ena and plot graphs for all iterations.
    train_loss_list = []
    valid_loss_list = []

    # Initialize the model and move to the computation device.
    model = create_model(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.\n")
    
    # Get the model parameters.
    params = [p for p in model.parameters() if p.requires_grad]
    # Define the optimizer.
    # optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    optimizer = torch.optim.AdamW(params, lr=0.0001, weight_decay=0.0005)

    # LR will be zero as we approach `steps` number of epochs each time.
    # If `steps = 5`, LR will slowly reduce to zero every 5 epochs.
    steps = NUM_EPOCHS + 25

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=steps,
        T_mult=1,
        verbose=True
    )

    # Initialize SaveBestModel class
    save_best_model = SaveBestModel()

    for epoch in range(NUM_EPOCHS):
        train_loss_hist.reset()

        _, batch_loss_list = train_one_epoch(
            model, 
            optimizer, 
            train_loader, 
            DEVICE, 
            epoch, 
            train_loss_hist,
            print_freq=100,
            scheduler=scheduler
        )

        _, val_loss_list = evaluate(model, valid_loader, DEVICE, CLASSES)

        # Add the current epoch's batch-wise lossed to the `train_loss_list`.
        train_loss_list.extend(batch_loss_list)
        valid_loss_list.extend(valid_loss_list)

         # save the best model till now if we have the least loss in the...
        # ... current epoch
        """ save_best_model(
            valid_loss_list, epoch, model, optimizer
        ) """
        # save the current epoch model
        save_model(epoch, model, optimizer)
        # save loss plot
        save_loss_plot_2(OUT_DIR, train_loss_list,valid_loss_list)
    
    total_time = time.time() - total_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("TOTAL TIME: "+total_time_str)
        


