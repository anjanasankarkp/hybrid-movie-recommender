import json
from abc_optimizer import ArtificialBeeColony

WEIGHTS_FILE = "abc_weights.json"

def train_and_save_weights(num_bees=20, iterations=30):

    colony = ArtificialBeeColony(
        num_bees=num_bees,
        max_iterations=iterations
    )

    best_weights, best_fitness = colony.optimize()

    weights = {
        "knn": float(best_weights[0]),
        "svd": float(best_weights[1]),
        "content": float(best_weights[2]),
        "fitness": float(best_fitness)
    }

    with open(WEIGHTS_FILE, "w") as f:
        json.dump(weights, f, indent=2)

    print("\nSaved ABC weights to", WEIGHTS_FILE)
    print(weights)

    return weights

if __name__ == "__main__":
    train_and_save_weights()