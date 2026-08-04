import pandas as pd
import numpy as np
import pickle
import os

ratings = pd.read_csv(
    "data/ml-100k/u.data", sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"]
)
movies = pd.read_csv(
    "data/ml-100k/u.item", sep="|", encoding="latin-1",
    header=None, usecols=[0, 1], names=["movie_id", "title"]
)

user_ids = ratings.user_id.unique()
movie_ids = ratings.movie_id.unique()

user_to_idx = {u: i for i, u in enumerate(user_ids)}
movie_to_idx = {m: i for i, m in enumerate(movie_ids)}

movie_id_to_title = dict(zip(movies.movie_id, movies.title))
title_to_movie_id = dict(zip(movies.title, movies.movie_id))

n_users = len(user_ids)
n_movies = len(movie_ids)


class SGDMomentumSVD:

    def __init__(self, n_factors=50, lr=0.005, reg=0.02, momentum=0.9, n_epochs=15):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.momentum = momentum
        self.n_epochs = n_epochs
        self.history = []

    def fit(self, ratings_df):

        rng = np.random.default_rng(42)

        self.P = rng.normal(0, 0.1, (n_users, self.n_factors))
        self.Q = rng.normal(0, 0.1, (n_movies, self.n_factors))
        self.b_u = np.zeros(n_users)
        self.b_i = np.zeros(n_movies)
        self.global_mean = ratings_df.rating.mean()

        # Momentum velocity terms
        vP, vQ = np.zeros_like(self.P), np.zeros_like(self.Q)
        vbu, vbi = np.zeros_like(self.b_u), np.zeros_like(self.b_i)

        data = ratings_df[["user_id", "movie_id", "rating"]].values

        for epoch in range(self.n_epochs):

            np.random.shuffle(data)
            sq_error_sum = 0

            for u_raw, m_raw, r in data:

                u = user_to_idx[u_raw]
                i = movie_to_idx[m_raw]

                pred = self.global_mean + self.b_u[u] + self.b_i[i] + np.dot(self.P[u], self.Q[i])
                error = r - pred
                sq_error_sum += error ** 2

                grad_bu = -(error - self.reg * self.b_u[u])
                grad_bi = -(error - self.reg * self.b_i[i])
                grad_P = -(error * self.Q[i] - self.reg * self.P[u])
                grad_Q = -(error * self.P[u] - self.reg * self.Q[i])

                # Momentum update
                vbu[u] = self.momentum * vbu[u] - self.lr * grad_bu
                vbi[i] = self.momentum * vbi[i] - self.lr * grad_bi
                vP[u] = self.momentum * vP[u] - self.lr * grad_P
                vQ[i] = self.momentum * vQ[i] - self.lr * grad_Q

                self.b_u[u] += vbu[u]
                self.b_i[i] += vbi[i]
                self.P[u] += vP[u]
                self.Q[i] += vQ[i]

            rmse = np.sqrt(sq_error_sum / len(data))
            self.history.append(rmse)
            print(f"Epoch {epoch+1}/{self.n_epochs}  RMSE={rmse:.4f}")

        return self

    def predict(self, user_id, movie_id):
        if user_id not in user_to_idx or movie_id not in movie_to_idx:
            return self.global_mean
        u, i = user_to_idx[user_id], movie_to_idx[movie_id]
        pred = self.global_mean + self.b_u[u] + self.b_i[i] + np.dot(self.P[u], self.Q[i])
        return float(np.clip(pred, 1, 5))


MODEL_PATH = "sgd_momentum_model.pkl"
model = None

def train_model():
    global model

    shuffled = ratings.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(shuffled) * 0.8)
    train_df = shuffled.iloc[:split_idx]
    test_df = shuffled.iloc[split_idx:]

    model = SGDMomentumSVD(n_factors=50, lr=0.005, reg=0.02, momentum=0.9, n_epochs=15)
    model.fit(train_df)

    # Held-out RMSE — the honest number
    test_errors = []
    for _, row in test_df.iterrows():
        pred = model.predict(row.user_id, row.movie_id)
        test_errors.append((row.rating - pred) ** 2)

    test_rmse = np.sqrt(np.mean(test_errors))
    print(f"\nHeld-out Test RMSE: {test_rmse:.4f}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    return model

def get_model():
    global model
    if model is not None:
        return model
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    else:
        model = train_model()
    return model

def recommend_sgd_momentum(user_id, movie_title, top_n=10):
    m = get_model()
    if movie_title not in title_to_movie_id:
        return []
    exclude_id = title_to_movie_id[movie_title]

    predictions = [
        (movie_id_to_title[mid], m.predict(user_id, mid))
        for mid in movie_ids if mid != exclude_id
    ]
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:top_n]


if __name__ == "__main__":
    train_model()
    recs = recommend_sgd_momentum(1, "Toy Story (1995)")
    print("\nTop Recommendations (SGD + Momentum)\n")
    for title, score in recs:
        print(f"{title}  ({score:.2f})")