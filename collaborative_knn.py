import pandas as pd

from sklearn.neighbors import NearestNeighbors

# ---------------------------------------------------
# Load MovieLens 100K
# ---------------------------------------------------

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

# ---------------------------------------------------
# Merge ratings with titles
# ---------------------------------------------------

movie_data = pd.merge(
    ratings,
    movies,
    on="movie_id"
)

# ---------------------------------------------------
# Keep popular movies only
# ---------------------------------------------------

movie_stats = movie_data.groupby("title")["rating"].count()

popular_movies = movie_stats[movie_stats >= 50].index

movie_data = movie_data[
    movie_data["title"].isin(popular_movies)
]

# ---------------------------------------------------
# User-Movie Matrix
# ---------------------------------------------------

movie_matrix = movie_data.pivot_table(
    index="title",
    columns="user_id",
    values="rating"
)

movie_matrix.fillna(0, inplace=True)

# ---------------------------------------------------
# Train KNN
# ---------------------------------------------------

model_knn = NearestNeighbors(
    metric="cosine",
    algorithm="brute"
)

model_knn.fit(movie_matrix)

# ---------------------------------------------------
# Recommendation Function
# ---------------------------------------------------

def knn_recommend(movie_title, top_n=10):

    if movie_title not in movie_matrix.index:

        return []

    movie_vector = movie_matrix.loc[movie_title].values.reshape(1, -1)

    distances, indices = model_knn.kneighbors(
        movie_vector,
        n_neighbors=top_n + 1
    )

    recommendations = []

    for distance, index in zip(
        distances.flatten()[1:],
        indices.flatten()[1:]
    ):

        title = movie_matrix.index[index]

        similarity = 1 - distance

        recommendations.append(
            (
                title,
                float(similarity)
            )
        )

    return recommendations

# ---------------------------------------------------
# Test
# ---------------------------------------------------

if __name__ == "__main__":

    movie = "Toy Story (1995)"

    recs = knn_recommend(movie)

    print("\nKNN Recommendations\n")

    for i, (title, score) in enumerate(recs, 1):

        print(f"{i}. {title} ({score:.3f})")