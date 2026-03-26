from pathlib import Path

MOTIFS_FILE = Path(
    "src/integrated_hessians/simulation/simple_simulation/simple_motifs.pfm"
)
SEQLEN = 100
BATCH_SIZE = 1000
EPOCHS = 100
LR = 1e-3
L2_WEIGHT_DECAY = 1e-5
TRAIN_DATA = Path("data/simple_simulation/100k.json")
OUT_BEST_MODEL = "data/simple_simulation/model_best.pth"
OUT_BEST_MODEL_EVAL = "data/simple_simulation/model_best_evaluation.json"
SEQLEN = 100
TEST_DATA = Path("data/simple_simulation/1k_test.json")
TEST_OUTPUT = Path("src/integrated_hessians/simulation/test/")
