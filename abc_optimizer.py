import random
import numpy as np

from collaborative_knn import knn_recommend
from svd_model import recommend_svd
from content_based import content_recommend

class ArtificialBeeColony:

    def __init__(
        self,
        num_bees=20,
        max_iterations=50,
        scout_limit=5
    ):

        self.num_bees = num_bees
        self.max_iterations = max_iterations
        self.scout_limit = scout_limit

        # Population of bees
        self.population = []

        # Fitness of every bee
        self.fitness = []

        # Number of unsuccessful attempts
        self.trial = []

        # Best solution found
        self.best_solution = None

        self.best_fitness = -float("inf")

    # -----------------------------------
    # Create one random solution
    # -----------------------------------

    def random_solution(self):

        weights = np.random.uniform(0.1, 1.0, 3)  # floor of 0.1 each

        weights = weights / np.sum(weights)

        return weights
        
    # -----------------------------------
    # Create initial population
    # -----------------------------------

    def initialize(self):

        self.population = []

        self.fitness = []

        self.trial = []

        for _ in range(self.num_bees):

            self.population.append(
                self.random_solution()
            )

            self.trial.append(0)

        print("\nInitial population created.")

        print("Number of bees :", len(self.population))

        
    # -----------------------------------
    # Display population
    # -----------------------------------

    def show_population(self):

        print()

        print("Initial Bee Population")

        print()

        for i, bee in enumerate(self.population, start=1):

            print(

                f"Bee {i}: "

                f"KNN={bee[0]:.3f} "

                f"SVD={bee[1]:.3f} "

                f"CONTENT={bee[2]:.3f}"

            )

    def fitness_function(
        self,
        weights,
        movie_name="Toy Story (1995)"
    ):

        w_knn, w_svd, w_content = weights

        scores = {}

        # -------------------------
        # KNN
        # -------------------------

        knn = knn_recommend(movie_name, top_n=10)

        for movie, similarity in knn:

            scores[movie] = scores.get(movie, 0) + similarity * w_knn


        # -------------------------
        # SVD
        # -------------------------

        svd = recommend_svd(
            user_id=1,
            movie_title=movie_name,
            top_n=10
        )

        for movie, rating in svd:

            # normalize rating to 0-1
            normalized = (rating - 1)/4

            scores[movie] = scores.get(movie, 0) + normalized * w_svd


        # -------------------------
        # Content
        # -------------------------

        clean_title = movie_name.split(" (")[0]

        content = content_recommend(
            clean_title,
            top_n=10
        )

        for movie, similarity in content:

            scores[movie] = scores.get(movie, 0) + similarity * w_content


        if len(scores) == 0:
            return 0

        # ----------------------------------------------------
        # Better Fitness (penalizes ignoring any source)
        # ----------------------------------------------------

        average_score = np.mean(list(scores.values()))

        unique_movies = len(scores)

        overlap = len(
            set(m for m, _ in knn) |
            set(m for m, _ in svd) |
            set(m for m, _ in content)
        )

        diversity = unique_movies / 30.0

        overlap_score = overlap / 10.0

        # NEW: reward using all three sources meaningfully.
        # This is maximized (=1) when weights are equal (0.33 each)
        # and drops toward 0 as any single weight dominates.
        balance_score = 1 - np.std(weights) / 0.4714  
        # 0.4714 = std of the most extreme case [1,0,0], used to normalize to ~0-1

        fitness = (
            0.40 * average_score +
            0.20 * diversity +
            0.20 * overlap_score +
            0.20 * balance_score
        )

        return fitness

    # -----------------------------------
    # Evaluate all bees
    # -----------------------------------

    def evaluate_population(self):

        self.fitness = []

        for bee in self.population:

            score = self.fitness_function(bee)

            self.fitness.append(score)

        print()

        print("Fitness Values")

        print()

        for i, score in enumerate(self.fitness, start=1):

            print(f"Bee {i}: {score:.4f}")

       # -----------------------------------
    # Employed Bee Phase
    # -----------------------------------

    def employed_bees(self):

        print("\nRunning Employed Bee Phase...\n")

        for i in range(self.num_bees):

            current = self.population[i].copy()

            # Choose another random bee
            k = random.randint(0, self.num_bees - 1)

            while k == i:
                k = random.randint(0, self.num_bees - 1)

            partner = self.population[k]

            # Random dimension (0=KNN,1=SVD,2=Content)
            j = random.randint(0, 2)

            # Random number between -1 and 1
            phi = random.uniform(-1, 1)

            # Create neighbour solution
            candidate = current.copy()

            candidate[j] = current[j] + phi * (
                current[j] - partner[j]
            )

            # Keep weights between 0 and 1
            candidate = np.clip(candidate, 0, None)

            # Normalize
            if candidate.sum() == 0:
                candidate = self.random_solution()
            else:
                candidate = candidate / candidate.sum()

            # Evaluate
            current_fitness = self.fitness[i]

            candidate_fitness = self.fitness_function(candidate)

            # Greedy Selection
            if candidate_fitness > current_fitness:

                self.population[i] = candidate

                self.fitness[i] = candidate_fitness

                self.trial[i] = 0

                print(
                    f"Bee {i+1} improved "
                    f"{current_fitness:.6f}"
                    f" -> "
                    f"{candidate_fitness:.6f}"
                )

            else:

                self.trial[i] += 1


    # -----------------------------------
    # Onlooker Bee Phase
    # -----------------------------------

    def onlooker_bees(self):

        print("\nRunning Onlooker Bee Phase...\n")

        fitness = np.array(self.fitness)

        if fitness.sum() == 0:
            probabilities = np.ones(self.num_bees) / self.num_bees
        else:
            probabilities = fitness / fitness.sum()

        for _ in range(self.num_bees):

            i = np.random.choice(
                range(self.num_bees),
                p=probabilities
            )

            current = self.population[i].copy()

            k = random.randint(0, self.num_bees - 1)

            while k == i:
                k = random.randint(0, self.num_bees - 1)

            partner = self.population[k]

            j = random.randint(0, 2)

            phi = random.uniform(-1, 1)

            candidate = current.copy()

            candidate[j] = current[j] + phi * (
                current[j] - partner[j]
            )

            candidate = np.clip(candidate, 0, None)

            if candidate.sum() == 0:

                candidate = self.random_solution()

            else:

                candidate = candidate / candidate.sum()

            candidate_fitness = self.fitness_function(candidate)

            if candidate_fitness > self.fitness[i]:

                print(
                    f"Bee {i+1} improved "
                    f"{self.fitness[i]:.4f} -> "
                    f"{candidate_fitness:.4f}"
                )

                self.population[i] = candidate

                self.fitness[i] = candidate_fitness

                self.trial[i] = 0

            else:

                self.trial[i] += 1

    # -----------------------------------
    # Scout Bee Phase
    # -----------------------------------

    def scout_bees(self):

        print("\nRunning Scout Bee Phase...\n")

        for i in range(self.num_bees):

            if self.trial[i] >= self.scout_limit:

                print(f"Bee {i+1} became a Scout Bee.")

                # Create a completely new random solution
                self.population[i] = self.random_solution()

                # Evaluate it
                self.fitness[i] = self.fitness_function(
                    self.population[i]
                )

                # Reset trial counter
                self.trial[i] = 0   

    def optimize(self, iterations=None):

        if iterations is None:
            iterations = self.max_iterations

        # Create the initial bee population
        self.initialize()

        # Display the population
        self.show_population()

        # Evaluate the population
        self.evaluate_population()

        best_index = np.argmax(self.fitness)

        best_solution = self.population[best_index].copy()

        best_fitness = self.fitness[best_index]

        for iteration in range(iterations):

            print("\n===================================")
            print(f"Iteration {iteration + 1}")
            print("===================================")

            self.employed_bees()

            self.onlooker_bees()

            self.scout_bees()

            #Recalculate fitness of all bees
            self.evaluate_population()

            current_best = np.argmax(self.fitness)

            if self.fitness[current_best] > best_fitness:

                best_fitness = self.fitness[current_best]

                best_solution = self.population[current_best].copy()

            print()

            print(f"Current Best Fitness : {best_fitness:.4f}")

            print(
                f"Weights -> "
                f"KNN={best_solution[0]:.3f}, "
                f"SVD={best_solution[1]:.3f}, "
                f"CONTENT={best_solution[2]:.3f}"
            )

        return best_solution, best_fitness
             

if __name__ == "__main__":

    colony = ArtificialBeeColony(
        num_bees=20,
        max_iterations=50
    )

    best_weights, best_score = colony.optimize()

    print("\n===================================")
    print("ABC Optimization Finished")
    print("===================================\n")

    print(f"KNN Weight     : {best_weights[0]:.4f}")
    print(f"SVD Weight     : {best_weights[1]:.4f}")
    print(f"Content Weight : {best_weights[2]:.4f}")

    print(f"\nBest Fitness : {best_score:.4f}")