import random
import pandas as pd

from collaborative_knn import knn_recommend
from svd_model import recommend_svd
from content_based import content_recommend
from hybrid_recommender import ABC_BASE_WEIGHTS

ratings = pd.read_csv(
    "data/ml-100k/u.data",
    sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"]
)

movies = pd.read_csv(
    "data/ml-100k/u.item",
    sep="|",
    encoding="latin-1",
    header=None,
    usecols=[0, 1],
    names=["movie_id", "title"]
)
movie_id_to_title = dict(zip(movies.movie_id, movies.title))

def get_relevant_movies(user_id):

    user_data = ratings[
        ratings["user_id"] == user_id
    ]

    relevant = user_data[
        user_data["rating"] >= 4
    ]

    relevant_titles = set(
        movie_id_to_title.get(mid) for mid in relevant["movie_id"]
    )

    return relevant_titles

def hybrid_recommend(
    user_id,
    movie_name,
    weights,
    top_n=10
):

    w_knn, w_svd, w_content = weights

    scores = {}

    # -----------------------
    # KNN
    # -----------------------

    for movie, similarity in knn_recommend(
        movie_name,
        top_n=top_n
    ):

        scores[movie] = scores.get(movie, 0) + (
            similarity * w_knn
        )

    # -----------------------
    # SVD
    # -----------------------

    for movie, rating in recommend_svd(
        user_id=user_id,
        movie_title=movie_name,
        top_n=top_n
    ):

        scores[movie] = scores.get(movie, 0) + (
            (rating / 5.0) * w_svd
        )

    # -----------------------
    # Content
    # -----------------------

    clean_title = movie_name.split(" (")[0]

    for movie, similarity in content_recommend(
        clean_title,
        top_n=top_n
    ):

        scores[movie] = scores.get(movie, 0) + (
            similarity * w_content
        )

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked[:top_n]


def precision_at_k(
    user_id,
    movie_name,
    weights,
    k=10
):

    recommendations = hybrid_recommend(
        user_id,
        movie_name,
        weights,
        top_n=k
    )

    relevant_movies = get_relevant_movies(user_id)

    if len(relevant_movies) == 0:
        return 0

    hits = 0

    for movie, score in recommendations:

        if movie in relevant_movies:
            hits += 1

    return hits / k

def compare_weight_strategies(user_id, movie_name, k=10):

    strategies = {
        "SVD-Only (naive)":  [0.0, 1.0, 0.0],
        "KNN-Only (naive)":  [1.0, 0.0, 0.0],
        "Uniform":           [0.33, 0.33, 0.34],
        "ABC Optimized":     list(ABC_BASE_WEIGHTS.values()),
    }

    print(f"\nComparing strategies for user {user_id}, movie '{movie_name}'\n")

    for name, weights in strategies.items():
        score = precision_at_k(user_id, movie_name, weights, k)
        print(f"{name:20} Precision@{k} = {score:.3f}")

if __name__ == "__main__":

    test_cases = [
        (1, "Toy Story (1995)"),
        (5, "Star Wars (1977)"),
        (10, "Fargo (1996)"),
        (15, "Pulp Fiction (1994)"),
        (22, "GoldenEye (1995)"),
    ]

    
    strategies = {
        "SVD-Only (naive)":  [0.0, 1.0, 0.0],
        "KNN-Only (naive)":  [1.0, 0.0, 0.0],
        "Uniform":           [0.33, 0.33, 0.34],
        "ABC Optimized":     list(ABC_BASE_WEIGHTS.values()),
    }

    results = {name: [] for name in strategies}

    for user_id, movie_name in test_cases:

        print(f"\n--- user {user_id}, '{movie_name}' ---")

        for name, weights in strategies.items():

            score = precision_at_k(user_id, movie_name, weights, k=10)

            results[name].append(score)

            print(f"{name:20} Precision@10 = {score:.3f}")

    print("\n=== Average Precision@10 across all test cases ===\n")

    for name, scores in results.items():

        avg = sum(scores) / len(scores)

        print(f"{name:20} Avg = {avg:.3f}")