#Directory
MODELS_DIR = "model"
MODEL_FILE = f"{MODELS_DIR}/aksara_sasak_model.h5"
JSON_FILE = f"{MODELS_DIR}/class_indices.json"

OUTPUT_DIR = "data/preprocessing"
TRAIN_DIR = f"{OUTPUT_DIR}/train"
VAL_DIR = f"{OUTPUT_DIR}/valid"

# TRAINING PARAMETERS
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
