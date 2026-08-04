import pandas as pd
import json
import os

from collaborative_knn import knn_recommend
from svd_model import recommend_svd
from content_based import content_recommend

from fuzzy_profile import (user_category, fuzzy_profile, viewing_pattern_profile)

ABC_WEIGHTS_PATH = "abc_weights.json"

def load_abc_weights():
    if os.path.exists(ABC_WEIGHTS_PATH):
        with open(ABC_WEIGHTS_PATH) as f:
            data = json.load(f)
            return {"knn": data["knn"], "svd": data["svd"], "content": data["content"]}
    # fallback if ABC hasn't been run yet
    return {"knn": 0.33, "svd": 0.34, "content": 0.33}

ABC_BASE_WEIGHTS = load_abc_weights()

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

movie_data = pd.merge(ratings, movies, on="movie_id")

rating_count = movie_data.groupby("title")["rating"].count()

def get_personalized_weights(user_id):

    profile = fuzzy_profile(user_id)

    result = user_category(profile)

    viewing = viewing_pattern_profile(user_id)

    print("Viewing Pattern Category:", viewing["primary_viewing_category"])
    print("Viewing Scores:", viewing["viewing_scores"])

    print("\n==============================")
    print("FUZZY USER PROFILE")
    print("==============================")

    print("Primary Category:", result["primary_category"])

    print("\nSecondary Interests:")

    for item in result["secondary_categories"]:
        print(
            f"{item['category']:25}"
            f"{item['membership']:.2f}"
        )

    membership = result["membership"]

    knn = 0
    svd = 0
    content = 0

    for genre, score in membership.items():

        if genre == "Action":
            content += score
            knn += score * 0.3

        elif genre == "Adventure":
            content += score
            knn += score * 0.2

        elif genre == "Animation":
            content += score

        elif genre == "Science Fiction":
            content += score

        elif genre == "Fantasy":
            content += score

        elif genre == "Drama":
            svd += score

        elif genre == "Romance":
            svd += score

        elif genre == "History":
            svd += score

        elif genre == "Documentary":
            svd += score

        elif genre == "Comedy":
            knn += score * 0.5
            svd += score * 0.5

        elif genre == "Crime":
            knn += score

        elif genre == "Mystery":
            knn += score

        elif genre == "Thriller":
            knn += score

        elif genre == "Family":
            content += score * 0.5
            svd += score * 0.5

        elif genre == "Horror":
            knn += score

        elif genre == "Music":
            svd += score

        elif genre == "War":
            svd += score

        elif genre == "Western":
            knn += score

        elif genre == "Foreign":
            content += score

    total = knn + svd + content

    fuzzy_weights = {
        "knn": knn / total,
        "svd": svd / total,
        "content": content / total
    }

    # How much to trust the ABC global optimum vs this user's fuzzy profile
    ABC_MIX = 0.6
    FUZZY_MIX = 1 - ABC_MIX

    blended = {
        key: ABC_MIX * ABC_BASE_WEIGHTS[key] + FUZZY_MIX * fuzzy_weights[key]
        for key in ["knn", "svd", "content"]
    }

    blend_total = sum(blended.values())

    weights = {
        key: round(value / blend_total, 3)
        for key, value in blended.items()
    }

    print("\nDynamic Fuzzy Weights")
    print("----------------------")
    print(f"KNN      : {weights['knn']:.3f}")
    print(f"SVD      : {weights['svd']:.3f}")
    print(f"Content  : {weights['content']:.3f}")

    return weights

# ===================================================
# Hybrid Recommendation Function
# ===================================================

def hybrid_recommend(movie_name, user_id=1, top_n=10):

    weights = get_personalized_weights(user_id)

    KNN_WEIGHT = weights["knn"]
    SVD_WEIGHT = weights["svd"]
    CONTENT_WEIGHT = weights["content"]

    print("\nPersonalized Weights")
    print(weights)

    scores = {}

    # -----------------------------------------------
    # KNN Recommendations
    # -----------------------------------------------

    knn_results = knn_recommend(
        movie_name,
        top_n=top_n
    )

    for movie, similarity in knn_results:

        scores[movie] = scores.get(movie, 0) + (
            similarity * KNN_WEIGHT
        )

    # -----------------------------------------------
    # SVD Recommendations
    # -----------------------------------------------

    svd_results = recommend_svd(
        user_id=user_id,
        movie_title=movie_name,
        top_n=top_n
    )

    for movie, rating in svd_results:

        normalized_rating = (rating - 1) / 4

        scores[movie] = scores.get(movie, 0) + (
            normalized_rating * SVD_WEIGHT
        )

    # -----------------------------------------------
    # Content-Based Recommendations
    # -----------------------------------------------

    clean_title = movie_name.split(" (")[0]

    content_results = content_recommend(
        clean_title,
        top_n=top_n
    )

    for movie, similarity in content_results:

        scores[movie] = scores.get(movie, 0) + (
            similarity * CONTENT_WEIGHT
        )

    # -----------------------------------------------
    # Convert to DataFrame
    # -----------------------------------------------

    recommendations = pd.DataFrame(
        scores.items(),
        columns=["Movie", "Score"]
    )

    recommendations = recommendations.sort_values(
        by="Score",
        ascending=False
    )

    recommendations = recommendations.head(top_n)

    recommendations["rating_count"] = (
        recommendations["Movie"]
        .map(rating_count)
        .fillna(0)
    )

    recommendations = recommendations.set_index("Movie")

    return recommendations



