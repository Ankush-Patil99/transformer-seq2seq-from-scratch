# config.py

# Model hyperparameters
D_MODEL = 384
NUM_HEADS = 6
NUM_LAYERS = 4
D_FF = 768
DROPOUT = 0.1
MAX_LEN = 200

# Training
LR = 3e-4
EPOCHS = 5
LABEL_SMOOTHING = 0.05
BEAM_WIDTH = 3

# Special tokens
PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3
