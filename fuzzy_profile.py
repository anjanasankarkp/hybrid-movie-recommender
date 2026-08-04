import pandas as pd
import ast

ratings = pd.read_csv(
    "data/ml-100k/u.data",
    sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"]
)

movies = pd.read_csv(
    "data/movies_metadata.csv",
    low_memory=False
)

links = pd.read_csv(
    "data/links.csv"
)

movies = movies[
    [
        "id",
        "title",
        "genres"
    ]
]

movies = movies.dropna(subset=["id"])

movies = movies[movies["id"].str.isnumeric()]

movies["id"] = movies["id"].astype(int)

links = links.dropna()

links["tmdbId"] = links["tmdbId"].astype(int)

movie_map = pd.merge(
    links,
    movies,
    left_on="tmdbId",
    right_on="id"
)

def parse_genres(text):

    try:

        data = ast.literal_eval(text)

        return [
            item["name"]
            for item in data
        ]

    except:

        return []

movie_map["genres"] = movie_map["genres"].apply(parse_genres)

def build_user_profile(user_id):

    user = ratings[
        ratings.user_id == user_id
    ]

    user = user[user.rating >= 4]

    merged = pd.merge(
        user,
        movie_map,
        left_on="movie_id",
        right_on="movieId"
    )

    if merged.empty:
        return {}

    genre_count = {}

    for genres in merged["genres"]:

        for g in genres:

            genre_count[g] = genre_count.get(g, 0) + 1

    return genre_count

def fuzzy_scale(x, low, high, invert=False):
    """Squashes x into a 0-1 membership score between low and high."""
    if high == low:
        return 0.0
    val = (x - low) / (high - low)
    val = min(max(val, 0), 1)
    if invert:
        val = 1 - val
    return round(val, 2)


def viewing_pattern_profile(user_id):

    user_ratings = ratings[ratings.user_id == user_id]

    n_ratings = len(user_ratings)

    if n_ratings == 0:
        return {
            "primary_viewing_category": "Casual Watcher",
            "viewing_scores": {},
        }

    # How spread out (in days) are this user's ratings?
    timespan_days = (
        user_ratings.timestamp.max() - user_ratings.timestamp.min()
    ) / 86400
    timespan_days = max(timespan_days, 1)

    ratings_per_day = n_ratings / timespan_days

    # Genre diversity (reuses the movie_map you already built above)
    merged = pd.merge(
        user_ratings, movie_map,
        left_on="movie_id", right_on="movieId"
    )

    unique_genres = set()
    for genre_list in merged["genres"]:
        unique_genres.update(genre_list)

    diversity = len(unique_genres)

    # Fuzzy membership degrees (linguistic clusters)
    binge_score = fuzzy_scale(ratings_per_day, low=0.5, high=5)
    casual_score = fuzzy_scale(n_ratings, low=5, high=50, invert=True)
    enthusiast_score = fuzzy_scale(diversity, low=3, high=15)

    scores = {
        "Binge Watcher": binge_score,
        "Casual Watcher": casual_score,
        "Genre Enthusiast": enthusiast_score,
    }

    primary = max(scores, key=scores.get)

    return {
        "primary_viewing_category": primary,
        "viewing_scores": scores,
        "stats": {
            "n_ratings": n_ratings,
            "ratings_per_day": round(ratings_per_day, 2),
            "genre_diversity": diversity,
        }
    }

def fuzzy_profile(user_id):

    profile = build_user_profile(user_id)

    if not profile:
        return {}

    max_count = max(profile.values())

    fuzzy = {}

    for genre, count in profile.items():

        fuzzy[genre] = round(count / max_count, 2)

    return fuzzy

def user_category(fuzzy):

    if not fuzzy:
        return {
            "primary_category": "Casual Viewer",
            "secondary_categories": [],
            "membership": {}
        }

    category_map = {

        "Action": "Action Fan",
        "Adventure": "Adventure Explorer",
        "Animation": "Animation Fan",
        "Comedy": "Comedy Lover",
        "Crime": "Crime Fan",
        "Documentary": "Documentary Viewer",
        "Drama": "Drama Enthusiast",
        "Family": "Family Viewer",
        "Fantasy": "Fantasy Fan",
        "Foreign": "Foreign Cinema Lover",
        "History": "History Buff",
        "Horror": "Horror Fan",
        "Music": "Music Movie Lover",
        "Mystery": "Mystery Lover",
        "Romance": "Romance Lover",
        "Science Fiction": "Sci-Fi Fan",
        "Thriller": "Thriller Fan",
        "War": "War Movie Fan",
        "Western": "Western Fan"

    }

    ranked = sorted(
        fuzzy.items(),
        key=lambda x: x[1],
        reverse=True
    )

    primary_genre = ranked[0][0]

    primary_category = category_map.get(
        primary_genre,
        "Movie Lover"
    )

    secondary_categories = []

    for genre, score in ranked[1:]:

        if score >= 0.30:

            secondary_categories.append({
                "genre": genre,
                "category": category_map.get(
                    genre,
                    genre + " Fan"
                ),
                "membership": score
            })

    return {

        "primary_category": primary_category,

        "secondary_categories": secondary_categories,

        "membership": fuzzy

    }

if __name__ == "__main__":

    user_id = 1

    profile = fuzzy_profile(user_id)

    result = user_category(profile)

    print("\n==============================")
    print("FUZZY USER PROFILE")
    print("==============================\n")

    print("Genre Memberships:\n")

    for genre, value in sorted(
        result["membership"].items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"{genre:20} {value:.2f}")

    print("\nPrimary Category:")
    print(result["primary_category"])

    print("\nSecondary Interests:")

    if len(result["secondary_categories"]) == 0:
        print("None")
    else:

        for item in result["secondary_categories"]:

            print(
                f"{item['category']:25}"
                f"{item['membership']:.2f}"
            )   