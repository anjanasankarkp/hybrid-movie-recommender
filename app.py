from flask import Flask, render_template,request
from hybrid_recommender import hybrid_recommend
from train import movie_list
from omdb import get_movie_poster
from flask import jsonify
from urllib.parse import unquote

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])

def home():

    recommendations = None
    movie_name = ""
    matched_movie = ""
    posters = {}
    search_message = ""
    error_message = ""
    searched_movie = None

    if request.method == "POST":
        movie_name = request.form["movie"]

        matched_movie = movie_name

        recommendations = hybrid_recommend(
            movie_name=movie_name,
            user_id=1,
            top_n=10
        )

        if recommendations.empty:

            error_message = (
                f'No movies found matching "{movie_name}". '
                "Please try another title."
            )

        elif movie_name.lower() != matched_movie.lower():

            search_message = matched_movie
        
        if recommendations is not None:

            total = len(recommendations)

            #Movie searched by user
            searched_movie = get_movie_poster(matched_movie)

            for index, (title, row) in enumerate (recommendations.iterrows()):
                posters[title] = get_movie_poster(title)
                
                # Calculate FilmNexa Match automatically
                stars = max(
                    1,
                    5 - round(index * 4 / max(total - 1, 1))
                )

                recommendations.loc[title, "stars"] = stars

    return render_template("index.html",
                           recommendations = recommendations,
                           movie_name=movie_name,
                           matched_movie=matched_movie,
                           movie_list=movie_list,
                           posters=posters,
                           search_message=search_message,
                           error_message=error_message,
                           searched_movie=searched_movie)

@app.route("/search")
def search():

    query = request.args.get("q", "").lower()

    if not query:
        return jsonify([])

    results = []

    for movie in movie_list:

        if query in movie.lower():
            
            info = get_movie_poster(movie)

            results.append({

                "title":movie,
                "poster": info["poster"] if info else None
            })

        if len(results) == 8:
            break

    print(results)

    return jsonify(results)

@app.route("/movie/<path:title>")
def movie_details(title):

    title = unquote(title)

    movie = get_movie_poster(title)

    if movie is None:

        return "Movie not found", 404

    return render_template(
        "details.html",
        title=title,
        movie=movie
    )
    
if __name__ == "__main__":
    app.run(debug=True, port=5001)

