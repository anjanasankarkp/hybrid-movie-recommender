import requests
from config import OMDB_API_KEY
import re
from cache import movie_cache

def get_movie_poster(movie_title):

    if movie_title in movie_cache:
        return movie_cache[movie_title]

    title, year = split_title_year(movie_title)

    if title.endswith(", The"):
        title = "The " + title[:-5]

    elif title.endswith(", A"):
        title = "A " + title[:-3]

    elif title.endswith(", An"):
        title = "An " + title[:-4]
 
    url = f"https://www.omdbapi.com/?apikey={OMDB_API_KEY}&t={title}&y={year}"
    response = requests.get(url)
    data = response.json()
    if data.get("Response") != "True":
        return None
        
    poster = data.get("Poster")

    if poster == "N/A":
        poster = None
            
    movie = {

    "poster": poster,

    "year": data.get("Year", "Unknown"),

    "genre": data.get("Genre", "Unknown"),

    "imdb": data.get("imdbRating", "N/A"),

    "plot": data.get("Plot", "No plot available."),

    "actors": data.get("Actors", "Unknown"),

    "director": data.get("Director", "Unknown"),

    "runtime": data.get("Runtime", "Unknown"),

    "awards": data.get("Awards", "None"),

    "language": data.get("Language", "Unknown"),

    "country": data.get("Country", "Unknown"),

    "writer": data.get("Writer", "Unknown"),

    "votes": data.get("imdbVotes", "N/A")

    }

    movie_cache[movie_title] = movie

    return movie
  
def split_title_year(movie_title):

    match = re.match(r"(.+)\s\((\d{4})\)", movie_title)

    if match:
        title = match.group(1)
        year = match.group(2)
    else:
        title = movie_title
        year = ""

    return title, year       

