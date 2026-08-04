import pandas as pd

from surprise import Dataset
from surprise import Reader
from surprise import SVD


# -----------------------------
# Load Dataset
# -----------------------------

ratings = pd.read_csv(
    "data/ml-100k/u.data",
    sep="\t",
    names=[
        "user_id",
        "movie_id",
        "rating",
        "timestamp"
    ]
)

movies = pd.read_csv(
    "data/ml-100k/u.item",
    sep="|",
    encoding="latin-1",
    header=None,
    usecols=[0, 1],
    names=[
        "movie_id",
        "title"
    ]
)

reader = Reader(rating_scale=(1, 5))

data = Dataset.load_from_df(

    ratings[
        [
            "user_id",
            "movie_id",
            "rating"
        ]
    ],

    reader

)

trainset = data.build_full_trainset()


# -----------------------------
# Train SVD
# -----------------------------

svd = SVD(
    n_factors=100,
    n_epochs=20,
    random_state=42
)

svd.fit(trainset)


movie_id_to_title = dict(

    zip(

        movies.movie_id,

        movies.title

    )

)

title_to_movie_id = dict(

    zip(

        movies.title,

        movies.movie_id

    )

)


# -----------------------------
# Recommend
# -----------------------------

def recommend_svd(
    user_id,
    movie_title,
    top_n=10
):

    if movie_title not in title_to_movie_id:

        return []

    movie_id = title_to_movie_id[movie_title]

    predictions = []

    for raw_movie_id in movies.movie_id:

        if raw_movie_id == movie_id:

            continue

        prediction = svd.predict(
            uid=user_id,
            iid=raw_movie_id
        )

        predictions.append(

            (

                movie_id_to_title[raw_movie_id],

                prediction.est

            )

        )

    predictions.sort(

        key=lambda x: x[1],

        reverse=True

    )

    return predictions[:top_n]


# -----------------------------
# Test
# -----------------------------

if __name__ == "__main__":

    recs = recommend_svd(
    user_id=1,
    movie_title="Toy Story (1995)"
    )

    print()

    print("Top Recommendations")

    print()

    for title, score in recs:

        print(

            f"{title}  ({score:.2f})"

        )