import json

CONFIG_PATHS = {
    "simple": "src/integrated_hessians/simulation/configs/simple.json",
    "custom": "src/integrated_hessians/simulation/configs/custom.json",
    "random": "src/integrated_hessians/simulation/configs/random.json",
}
SIMULATE_SCRIPTS = {
    "simple": "src/integrated_hessians/simulation/create_simulation_data/simple.py",
    "custom": "src/integrated_hessians/simulation/create_simulation_data/custom_additive_and_interactive_effects.py",
    "random": "src/integrated_hessians/simulation/create_simulation_data/random_additive_and_interactive_effects.py",
}
SIM_NAMES = list(CONFIG_PATHS.keys())
CONFIGS = {}
for sim, path in CONFIG_PATHS.items():
    with open(path, "r") as f:
        CONFIGS[sim] = json.load(f)

rule all:
    input:
        [CONFIGS[s]["OUT_BEST_MODEL"] for s in SIM_NAMES],
        [CONFIGS[s]["OUT_BEST_MODEL_EVAL"] for s in SIM_NAMES],

# Instead of a normal for loop, we have a for loop with a function call. This prevents a bug where wrong configs being passed to scripts.
def make_rules(sim):
    rule:
        name: f"simulate_dataset_{sim}"
        input: CONFIG_PATHS[sim]
        output: CONFIGS[sim]["TRAIN_DATA"], CONFIGS[sim]["TEST_DATA"]
        shell: f"uv run {SIMULATE_SCRIPTS[sim]} {{input}}"

    rule:
        name: f"train_model_{sim}"
        input:
            CONFIGS[sim]["TRAIN_DATA"],
            CONFIGS[sim]["TEST_DATA"],
            config=CONFIG_PATHS[sim],
        output:
            CONFIGS[sim]["OUT_BEST_MODEL"],
            CONFIGS[sim]["OUT_BEST_MODEL_EVAL"],
        shell:
            f"uv run src/integrated_hessians/simulation/train_model.py {{input.config}}"

for sim in SIM_NAMES:
    make_rules(sim)
