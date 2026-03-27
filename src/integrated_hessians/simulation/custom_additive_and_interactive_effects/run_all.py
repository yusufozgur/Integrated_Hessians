from integrated_hessians.simulation.custom_additive_and_interactive_effects.create_simulation import (
    main as create_sim_data,
)
from integrated_hessians.simulation.custom_additive_and_interactive_effects.train_model import (
    main as train_model,
)
from integrated_hessians.simulation.custom_additive_and_interactive_effects.test_model import (
    main as test_model,
)

if __name__ == "__main__":
    create_sim_data()
    train_model()
    test_model()
