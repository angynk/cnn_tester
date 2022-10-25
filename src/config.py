import torch

# Number of Samples, increase / decrease according to GPU memory
BATCH_SIZE = 4 
# Resize the image for trainning and transforms INICIAL 960 540 
RESIZE_TO = 512
# Number of epochs to train for
NUM_EPOCHS = 200
NUM_WORKERS = 4

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# training images and XML files directory
TRAIN_DIR = 'dataset/train'
# validation images and XML files directory
VALID_DIR = 'dataset/valid'
# test images and XML files directory
TEST_DIR ='test_data/'

# classes: 0 index is reserved for background
CLASSES = [
    'crutches', 'person', 'push_wheelchair', 'walking_frame', 'wheelchair'
]

NUM_CLASSES = len(CLASSES)

# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False

# location to save model and plots
OUT_DIR = 'outputs'

# First Hyperparameters
HFLIP = 0.8
RSCALE = 0.5
RHSV = 0.8
RTRANSLATE = 0.4
RSHEAR = 0.5

CUTOUT = 0.5
RAUG = 0.8
AUGMENTATION = 0.8
