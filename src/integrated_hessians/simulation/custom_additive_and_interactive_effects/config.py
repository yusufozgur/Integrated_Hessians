from pathlib import Path

MOTIFS_FILE = Path(
    "src/integrated_hessians/simulation/custom_additive_and_interactive_effects/custom_motifs.pfm"
)
SEQLEN = 100
BATCH_SIZE = 1000
EPOCHS = 100
LR = 1e-3
L2_WEIGHT_DECAY = 1e-5
TRAIN_DATA = Path("data/custom_additive_and_interactive_effects/100k.json")
OUT_BEST_MODEL = "data/custom_additive_and_interactive_effects/model_best.pth"
OUT_BEST_MODEL_EVAL = (
    "data/custom_additive_and_interactive_effects/model_best_evaluation.json"
)
OUT_EXTRACTED_ADDITIVE_EFFECTS = Path(
    "data/custom_additive_and_interactive_effects/additive_effects_extracted_from_attributions.json"
)
OUT_EXTRACTED_INTERACTIVE_EFFECTS = Path(
    "data/custom_additive_and_interactive_effects/interactive_effects_extracted_from_integrated_hessians.json"
)
SEQLEN = 100
TEST_DATA = Path("data/custom_additive_and_interactive_effects/1k_test.json")
TEST_OUTPUT = Path(
    "src/integrated_hessians/simulation/custom_additive_and_interactive_effects/test/"
)
INTEGRATED_HESSIANS_SAMPLING_STEPS = 1
NUM_OF_ROWS_TESTED = 3
