from integrated_hessians.simulation.randomized_additive_and_interactive_effects.create_simulation import (
    main as create_sim_data,
)
from integrated_hessians.simulation.randomized_additive_and_interactive_effects.train_model import (
    main as train_model,
)
from integrated_hessians.simulation.randomized_additive_and_interactive_effects.test_model import (
    main as test_model,
)

from integrated_hessians.simulation.randomized_additive_and_interactive_effects.extract_rules import (
    main as extract_rules,
)

if __name__ == "__main__":
    create_sim_data()
    train_model()
    test_model()
    extract_rules()
