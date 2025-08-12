import torch

# ---------------------------------------
# 🔤 Character Set and Index Mapping
# ---------------------------------------
# ------------------------------
# 🔤 Word Bank for Sentence Construction
# ------------------------------
SEED_WORD = [
    # Document-related
    "passport", "citizen", "national", "identity", "visa", "residence", "permit", "license", "birth",
    "certificate", "security", "number", "expiry", "issue", "government", "immigration", "border",
    "embassy", "consulate", "republic", "kingdom", "authority", "hologram", "signature", "biometric",
    "fingerprint", "barcode", "qrcode", "seal", "dob", "dateofbirth",

    # Common first names
    "john", "mary", "peter", "linda", "david", "susan", "robert", "james", "jennifer", "michael",
    "william", "elizabeth", "daniel", "joseph", "emma", "olivia", "sophia", "ava", "isabella", "mia",

    # Common last names
    "smith", "johnson", "williams", "brown", "jones", "garcia", "miller", "davis", "rodriguez",
    "martinez", "hernandez", "lopez", "gonzalez", "wilson", "anderson", "thomas", "taylor", "moore",
    "jackson", "martin",

    # Places
    "vietnam", "canada", "japan", "france", "germany", "brazil", "india", "china", "italy", "australia",
    "hanoi", "toronto", "tokyo", "paris", "berlin", "rio", "delhi", "beijing", "rome", "sydney",
    "province", "city", "district", "village", "capital", "state", "territory",

    # ID number related
    "idnumber", "passportnumber", "cardnumber", "serialnumber", "registration", "document", "code",
    "identifier", "number", "num", "no", "ref", "reference", "mrz", "mrzcode"
]

CHARSET = (
    "abcdefghijklmnopqrstuvwxyz"
)


CHAR_TO_INDEX: dict[str, int] = {char: idx + 1 for idx, char in enumerate(CHARSET)}  # 0 = blank
INDEX_TO_CHAR: dict[int, str] = {idx + 1: char for idx, char in enumerate(CHARSET)}
BLANK_IDX: int = 0 # +1 for CTC blank token
NUM_CLASSES: int = len(CHARSET) + 1 # +1 for CTC blank token


# ------------------------------
# 🛠️ Training Configuration
# ------------------------------
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE     = 32
NUM_EPOCHS     = 50000
LR_SCHEDULE    = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
SAVE_EVERY     = 1

EARLY_STOPPING_PATIENCE = 10
VAL_CHECK_INTERVAL      = 1  # Validate every N epochs

TRAIN_DIR      = "data/train"
VAL_DIR        = "data/val"
PREDICT_DIR    = "data/predict"

MODEL_PATH          = "checkpoints/crnn_curr.pth"
BEST_MODEL_PATH     = "checkpoints/crnn_best.pth"

TARGET_HEIGHT = 80
TARGET_WIDTH  = 400

LOAD_CHECKPOINT = False


# ------------------------------
# 📁 Configuration
# ------------------------------
FONTS_DIR          = "fonts"
IMAGE_SIZE         = (TARGET_WIDTH, TARGET_HEIGHT)  # From config
CHARS              = "".join(CHARSET)  # Ensure consistency with CHARSET used in model