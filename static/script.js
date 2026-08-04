const movieInput = document.getElementById("movie");
const suggestions = document.getElementById("suggestions");

let timer;

// =======================
// Movie Search
// =======================

if (movieInput && suggestions) {

    movieInput.addEventListener("input", () => {

        clearTimeout(timer);

        timer = setTimeout(searchMovie, 300);

    });

}

async function searchMovie() {

    const query = movieInput.value.trim();

    if (query.length < 2) {

        suggestions.innerHTML = "";

        suggestions.style.display = "none";

        return;

    }

    try {

        const response = await fetch("/search?q=" + encodeURIComponent(query));

        const movies = await response.json();

        suggestions.innerHTML = "";

        movies.forEach(movie => {

            const div = document.createElement("div");

            div.className = "suggestion";

            div.innerHTML = `
                <img src="${movie.poster || '/static/no_poster.png'}">
                <span>${movie.title}</span>
            `;

            div.addEventListener("click", () => {

                movieInput.value = movie.title;

                suggestions.style.display = "none";

            });

            suggestions.appendChild(div);

        });

        suggestions.style.display = movies.length ? "block" : "none";

    }

    catch (error) {

        console.error("Search Error:", error);

    }

}

// =======================
// Hide Suggestions
// =======================

document.addEventListener("click", (event) => {

    if (
        movieInput &&
        suggestions &&
        !movieInput.contains(event.target) &&
        !suggestions.contains(event.target)
    ) {

        suggestions.style.display = "none";

    }

});

// =======================
// Discover Button Loading
// =======================

const form = document.getElementById("movieForm");
const button = document.getElementById("discoverBtn");

if (form && button) {

    form.addEventListener("submit", () => {

        button.innerHTML = `
            <span class="loader"></span>
            Finding Movies...
        `;

        button.disabled = true;

    });

}

// =======================
// Watchlist Button
// =======================

const watchButtons = document.querySelectorAll(".watch-btn");

watchButtons.forEach(button => {

    button.addEventListener("click", function () {

        this.classList.toggle("saved");

        if (this.classList.contains("saved")) {

            this.innerHTML = "❤️ Saved";

        }

        else {

            this.innerHTML = "❤️ Save to Watchlist";

        }

    });

});