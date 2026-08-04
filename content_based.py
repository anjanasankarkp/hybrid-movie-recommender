import pandas as pd
import ast

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# ----------------------------------------------------
# Load datasets
# ----------------------------------------------------

movies = pd.read_csv(
    "data/movies_metadata.csv",
    low_memory=False
)

credits = pd.read_csv(
    "data/credits.csv"
)

keywords = pd.read_csv(
    "data/keywords.csv"
)

# ----------------------------------------------------
# Keep required columns
# ----------------------------------------------------

movies = movies[
    [
        "id",
        "title",
        "overview",
        "genres"
    ]
]

movies = movies.dropna(subset=["id"])
movies = movies[movies["id"].str.isnumeric()]
movies["id"] = movies["id"].astype(int)

credits["id"] = credits["id"].astype(int)
keywords["id"] = keywords["id"].astype(int)

# ----------------------------------------------------
# Merge
# ----------------------------------------------------

movies = movies.merge(
    credits,
    on="id"
)

movies = movies.merge(
    keywords,
    on="id"
)

# ----------------------------------------------------
# Convert JSON text
# ----------------------------------------------------

def convert(text):

    try:

        obj = ast.literal_eval(text)

        return " ".join(
            item["name"]
            for item in obj
        )

    except:

        return ""

movies["genres"] = movies["genres"].apply(convert)
movies["keywords"] = movies["keywords"].apply(convert)
movies["cast"] = movies["cast"].apply(convert)
movies["crew"] = movies["crew"].apply(convert)

movies["overview"] = movies["overview"].fillna("")

# ----------------------------------------------------
# Create Tags
# ----------------------------------------------------

movies["tags"] = (

    movies["overview"]

    + " "

    + movies["genres"]

    + " "

    + movies["keywords"]

    + " "

    + movies["cast"]

    + " "

    + movies["crew"]

)

# ----------------------------------------------------
# TF-IDF
# ----------------------------------------------------

tfidf = TfidfVectorizer(
    stop_words="english"
)

tfidf_matrix = tfidf.fit_transform(
    movies["tags"]
)

indices = pd.Series(
    movies.index,
    index=movies["title"]
).drop_duplicates()

# ----------------------------------------------------
# Recommendation Function
# ----------------------------------------------------

def content_recommend(movie_title, top_n=10):

    if movie_title not in indices:

        return []

    idx = indices[movie_title]
    
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]

    similarity = linear_kernel(
        tfidf_matrix[idx:idx+1],
        tfidf_matrix
    ).flatten()

    similarity_scores = list(
        enumerate(similarity)
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    similarity_scores = similarity_scores[1:top_n+1]

    recommendations = []

    for index, score in similarity_scores:

        recommendations.append(

            (
                movies.iloc[index]["title"],
                float(score)
            )

        )

    return recommendations

# ----------------------------------------------------
# Test
# ----------------------------------------------------

if __name__ == "__main__":

    recommendations = content_recommend(
        "Toy Story",
        10
    )

    print()

    print("Content Recommendations")

    print()

    for title, score in recommendations:

        print(title, round(score, 3))