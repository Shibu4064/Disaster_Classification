from pathlib import Path


DATA_ROOT = Path("data/RetinaX")
TRAIN_CSV = DATA_ROOT / "train_data.csv"
VAL_CSV = DATA_ROOT / "test_data.csv"
IMG_DIR = DATA_ROOT / "images"
EMB_CACHE = DATA_ROOT / "label_embeddings.npy"

CHECKPOINT_PATH = Path("checkpoints/best_gatn_model.pt")

IMG_RESIZE = 320
IMG_CROP = 300
RETINA_MEAN = [0.357, 0.287, 0.226]
RETINA_STD = [0.172, 0.153, 0.161]

BATCH_SIZE = 32
NUM_WORKERS = 2
NUM_EPOCHS = 20
BASE_LR = 1e-4

GAT_HEADS = 4
GCN_HIDDEN = 1024
ALPHA = 1.0

BERT_MODEL = "bert-base-uncased"
