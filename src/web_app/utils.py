import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import linear_kernel


def write_anime_details(row, name):
    rating = np.round(row["average_rating_1"].values[0], 2)
    total_ratings = row['total_ratings'].values[0]
    synopsis = row['synopsis'].values[0]
    episodes = row['Episodes'].values[0]
    members = row['Members'].values[0]
    premier = row['Premiered'].values[0]
    favorites = row['Favorites'].values[0]
    duration = row['Duration'].values[0]
    url = get_google_search_url(name)

    st.write(
        "***{}*** :star: with ***{:,}*** Ratings ".format(rating, total_ratings))
    st.write(
        ":clapper:Episodes: ***{}***".format("Unknown" if episodes == 0 else episodes))
    st.write(":hourglass_flowing_sand:Duration: {}".format(
        duration if episodes.astype(int) >= 1 else None))
    st.write(":calendar:Premeired: {}".format(
        premier if premier != "Unknown" else None))
    st.write(":male-technologist:Viewers: ***{:,}***".format(members))
    st.write(":thumbsup: Favorites: ***{:,}***".format(favorites))
    st.write(f'<a href="{url}" target="_blank" style="color:black; text-decoration:none"> :black_right_pointing_triangle_with_double_vertical_bar: Watch Now</a>', unsafe_allow_html=True)
    st.divider()
    st.header("Storyline")
    st.write(synopsis)


def write_anime_details_wide(row, name):
    rating = np.round(row["average_rating_1"].values[0], 2)
    total_ratings = row['total_ratings'].values[0]
    synopsis = row['synopsis'].values[0]
    episodes = row['Episodes'].values[0]
    members = row['Members'].values[0]
    premier = row['Premiered'].values[0]
    favorites = row['Favorites'].values[0]
    duration = row['Duration'].values[0]
    url = get_google_search_url(name)

    st.write(f'<a href="{url}" target="_blank" style="color:black; text-decoration:none"> :black_right_pointing_triangle_with_double_vertical_bar: Watch Now</a>' + "&nbsp;"*10 + "***{}*** :star:".format(rating) + "&nbsp;"*10 + " :busts_in_silhouette:***{:,}***".format(total_ratings) + "&nbsp;"*10 + ":clapper:Episodes: ***{}***".format(
        "Unknown" if episodes == 0 else episodes) + "&nbsp;"*10 +
        ":hourglass_flowing_sand:{}".format(duration if episodes.astype(int) >= 1 else None) + "&nbsp;"*10 +
        ":calendar: {}".format(
        premier if premier != "Unknown" else None) + "&nbsp;"*10 +
        ":male-technologist: ***{:,}***".format(members) + "&nbsp;"*10 +
        ":thumbsup:  ***{:,}***".format(favorites), unsafe_allow_html=True)

    st.divider()
    st.header("Storyline")
    st.write(synopsis)


def get_google_search_url(anime_name):
    base_url = "https://www.google.com/search?q="
    search_query = anime_name.replace(' ', '+') + "+watch+online"
    return base_url + search_query


genre_emoji = {"Comedy": "Comedy :rolling_on_the_floor_laughing:", "Mecha": "Mecha :robot_face:", "Game": "Game :video_game:", "Military": "Military :gun:",
               "Vampire": "Vampire :vampire:", "Shounen": "Shounen :boy:", "Psychological": "Psychological :brain:", "Magic": "Magic :magic_wand:",
               "Slice of Life": "Slice of Life :cake:", "Drama": "Drama :film_projector:", "Supernatural": "Supernatural :ghost:",
               "Mystery": "Mystery :male-detective:", "School": "School :school:", "Romance": "Romance :sparkling_heart:",
               "Historical": "Historical :hourglass:", "Horror": "Horror :scream:", "Sports": "Sports :soccer:", "Sci-Fi": "Sci-Fi :rocket:",
               "Adventure": "Adventure :sunrise_over_mountains:", "Fantasy": "Fantasy :unicorn_face:", "Action": "Action :collision:"}


def get_weighted_recommendations(user_animes, user_ratings, cosine_sim, num_recommendations=10, indices=None, model_df=None):
    # Start with a zero matrix
    accumulator = np.zeros(cosine_sim.shape[0])
    weight_sum = np.zeros(cosine_sim.shape[0])

    # Add weighted similarity matrices
    for anime, rating in zip(user_animes, user_ratings):
        idx = indices[anime]
        accumulator += cosine_sim[idx] * rating
        weight_sum += cosine_sim[idx]

    # Normalize the accumulator
    mean_scores = np.where(weight_sum != 0, accumulator / weight_sum, 0)

    # Get recommendations based on the accumulated matrix
    sim_scores = [(i, mean_scores[i]) for i in range(len(mean_scores))]
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Considering the top 30 animes with the highest similarity score
    sim_scores = sim_scores[1:31]

    # Get the anime indices
    anime_indices = [i[0] for i in sim_scores]

    # Fetch the animes, excluding the ones that the user has already rated
    animes = model_df.iloc[anime_indices].copy()
    animes = animes[~animes['Name'].isin(user_animes)]

    # Filter out animes with average rating less than 5
    animes = animes[animes['average_rating_1'] >= 5]

    # Sort the animes by 'total_ratings' and get the top 'num_recommendations' animes
    top_animes = animes.sort_values(by='average_rating_1', ascending=False)[
        :num_recommendations]

    return list(top_animes['Name'].values)


def get_anime_name_based_on_story(user_input, tfidf=None, tfidf_matrix=None, model_df=None):
    user_vec = tfidf.transform([user_input])
    cosine_similarities = linear_kernel(user_vec, tfidf_matrix).flatten()

    # Get the top 10 most similar anime
    related_anime_indices = cosine_similarities.argsort()[:-11:-1]
    related_anime = model_df['Name'].iloc[related_anime_indices]
    return related_anime.values
