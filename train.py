import pandas as pd
import numpy as np
import difflib
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
#Loading dataset
ratings = pd.read_csv("data/ml-100k/u.data",sep="\t",names=["user_id", "movie_id", "rating", "timestamp"])

#print(ratings.head())

#print("\nNumber of rows and columns:")
#print(ratings.shape)

#print("\nColumn names:")
#print(ratings.columns)

#print("\nInformation:")
#ratings.info()

movies = pd.read_csv("data/ml-100k/u.item",sep = "|", encoding="latin-1",header =None, usecols=[0,1], names=["movie_id","title"])
#print("\nMovies Dataset:")
#print(movies.head())

#print("\nMovies Shape:")
#print(movies.shape)

#print("\nMovies Information:")
#movies.info()

#Merging two tables
movie_ratings = pd.merge(ratings, movies, on="movie_id")
#print("\nMerged Dataset:")
#print(movie_ratings.head())

#Start Analyzing the Data
#print("\nAverage Rating of Each Movie:")
average_rating = movie_ratings.groupby("title")["rating"].mean()
#print(average_rating.head(10))

#print("\nNumber of Ratings for Each Movie:")
rating_count = movie_ratings.groupby("title")["rating"].count()
#print(rating_count.head(10))

#First Analysis DataFrame
movie_stats = movie_ratings.groupby("title").agg(average_rating=("rating", 'mean'), rating_count=("rating", "count"))
#print("\nMovie Statistics:")
#print(movie_stats.head(10))

#Filter the Table
popular_movies = movie_stats[movie_stats["rating_count"] > 100]
#print("\nPopular Movies:")
#print(popular_movies.head(10))

#User-Movie Matrix
user_movie_matrix = movie_ratings.pivot_table(index="user_id", columns="title", values="rating")
#print("\nUser-Movie MAtrix:")
#print(user_movie_matrix.head())

#Build the Recommendation Engine
toy_story_ratings = user_movie_matrix["Toy Story (1995)"]
#print("\nToy Story Ratings:")
#print(toy_story_ratings.head(20))

#Correlation
toy_story_ratings = user_movie_matrix["Toy Story (1995)"]
similar_movies = user_movie_matrix.corrwith(toy_story_ratings)
#print("\nMovies Similar to Toy Story:")
#print(similar_movies.head(20))

#Makes Recommendation
corr_df = pd.DataFrame(similar_movies, columns=["correlation"])
corr_df.dropna(inplace=True)
corr_df = corr_df.join(movie_stats["rating_count"])
recommendations = corr_df[corr_df["rating_count"] > 100]
recommendations = recommendations.sort_values(by="correlation", ascending = False)
#print("\nRecommended Movies:")
#print(recommendations.head(10))

#Create a Recommendation Function
def recommend_movies(movie_name):
    
    if movie_name not in user_movie_matrix.columns:

        suggestions = difflib.get_close_matches(movie_name, user_movie_matrix.columns, n = 5, cutoff = 0.7)

        if suggestions:
            movie_name = suggestions[0]
        else:
            return None, None

    movie_ratings = user_movie_matrix[movie_name]

    similar_movies = user_movie_matrix.corrwith(movie_ratings)

    corr_df = pd.DataFrame(similar_movies, columns = ["correlation"])

    corr_df.dropna(inplace = True)

    corr_df = corr_df.join(movie_stats["rating_count"])

    recommendations = corr_df[corr_df["rating_count"] > 100]

    recommendations = recommendations.sort_values(by = "correlation", ascending = False)

    #Remove the selected movie
    if movie_name in recommendations.index:
        recommendations = recommendations.drop(movie_name)

    return recommendations.head(10), movie_name

#List of all movie titles for autocomplete
movie_list = sorted(user_movie_matrix.columns.tolist())

def search_movies(query):

    query = query.lower()

    results = []

    for movie in movie_list:
        if query in movie.lower():
            results.append(movie)

    return results[:10]

"""if __name__ == "__main__":
    
#Let user chosse the movie
    movie_name = input("Enter a movie name: ")

    recommendations = recommend_movies(movie_name)

    if recommendations is not None:

        print("\nTop 10 Recommendation Movies\n")

        for i, (title, row) in  enumerate(recommendations.iterrows(), start = 1):
            print(f"{i}. {title}")
            print(f" correlation:{row['correlation']:.3f}")
            print(f" Ratings:{int(row['rating_count'])}")
            print()"""


    

