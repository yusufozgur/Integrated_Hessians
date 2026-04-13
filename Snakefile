import json

# load config files

config_simple_path = "src/integrated_hessians/simulation/configs/simple.json"
with open(config_simple_path, "r") as f:
    config_simple = json.load(f)
config_custom_path = "src/integrated_hessians/simulation/configs/custom.json"
with open(config_custom_path, "r") as f:
    config_custom = json.load(f)
config_random_path = "src/integrated_hessians/simulation/configs/random.json"
with open(config_random_path, "r") as f:
    config_random = json.load(f)

# Run every rule
rule all:
    input:
        config_simple["OUT_BEST_MODEL"],
        config_simple["OUT_BEST_MODEL_EVAL"],
        config_custom["OUT_BEST_MODEL"],
        config_custom["OUT_BEST_MODEL_EVAL"],
        config_random["OUT_BEST_MODEL"],
        config_random["OUT_BEST_MODEL_EVAL"],


# Simple Simulation
rule simulate_dataset_simple:
    input: config_simple_path
    output:
        config_simple["TRAIN_DATA"],
        config_simple["TEST_DATA"]
    shell:
        "uv run src/integrated_hessians/simulation/create_simulation_data/simple.py {input}"

rule train_model_simple:
    input:
        config_simple["TRAIN_DATA"],
        config_simple["TEST_DATA"]
    output:
        config_simple["OUT_BEST_MODEL"],
        config_simple["OUT_BEST_MODEL_EVAL"]
    shell:
        "uv run src/integrated_hessians/simulation/train_model.py {config_simple_path}"


# Simulation with custom additive and interactive effects
rule simulate_dataset_custom:
    input: config_custom_path
    output:
        config_custom["TRAIN_DATA"],
        config_custom["TEST_DATA"]
    shell:
        "uv run src/integrated_hessians/simulation/create_simulation_data/custom_additive_and_interactive_effects.py {input}"


rule train_model_custom:
    input:
        config_custom["TRAIN_DATA"],
        config_custom["TEST_DATA"]
    output:
        config_custom["OUT_BEST_MODEL"],
        config_custom["OUT_BEST_MODEL_EVAL"]
    shell:
        "uv run src/integrated_hessians/simulation/train_model.py {config_custom_path}"

# Simulation with randomized additive and interactive effects
rule simulate_dataset_random:
    input: config_random_path
    output:
        config_random["TRAIN_DATA"],
        config_random["TEST_DATA"]
    shell:
        "uv run src/integrated_hessians/simulation/create_simulation_data/random_additive_and_interactive_effects.py {input}"


rule train_model_random:
    input:
        config_random["TRAIN_DATA"],
        config_random["TEST_DATA"]
    output:
        config_random["OUT_BEST_MODEL"],
        config_random["OUT_BEST_MODEL_EVAL"]
    shell:
        "uv run src/integrated_hessians/simulation/train_model.py {config_random_path}"
