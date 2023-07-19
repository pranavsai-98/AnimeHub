import base64
import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components as stc
import random
import os
from streamlit_option_menu import option_menu
from PIL import Image
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from utils import *


st.set_page_config(page_title="Home", page_icon=":rocket:", layout="wide")

st.title("AnimeHub :clapper:")

selected = option_menu(
    menu_title=None,
    options=['Home', 'Dashboard'],
    icons=["house", 'rocket-takeoff'],
    menu_icon="cast",
    default_index=0,
    orientation='horizontal',
)

if selected == 'Home':

    current_path = os.path.dirname(__file__)
    Thumbnail_folder = os.path.join(current_path, "../anime_images/")
    front_end_data = pd.read_csv(os.path.join(
        current_path, "../web_app/front_end_data.csv"))

    with open(os.path.join(current_path, '../web_app/home_page_anime_genre_2.pkl'), 'rb') as handle:
        hp_genre_anime = pickle.load(handle)

    genres = list(hp_genre_anime.keys())

    tabs = st.tabs(genres)

    for i, tab in enumerate(tabs):
        with tab:

            st.header(genre_emoji[genres[i]])
            cols = st.columns(5)

            for j, col in enumerate(cols):
                with col:
                    try:
                        image_name = hp_genre_anime[genres[i]][j]
                        image_file = "_".join(image_name.split(" ")) + ".jpg"
                        image_path = os.path.join(Thumbnail_folder, image_file)
                        image = Image.open(image_path)
                        st.write("***{}***".format(image_name))
                        with st.expander("Know more"):
                            row = front_end_data[front_end_data["Name"]
                                                 == image_name]
                            write_anime_details(row, image_name)

                        st.image(image, use_column_width=True)

                    except IndexError:
                        break

    st.write("#")

    st.divider()

    st.header("Get Personalized Anime Recommendations:gift:")

    st.write("#")

    model_df = front_end_data.copy(deep=True)
    model_df = model_df[model_df['total_ratings'] > 30000]
    model_df.reset_index(inplace=True, drop=True)

    tfidf = TfidfVectorizer(stop_words='english')

    model_df['synopsis'] = model_df['synopsis'].fillna('')
    tfidf_matrix = tfidf.fit_transform(model_df['synopsis'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    indices = pd.Series(model_df.index, index=model_df['Name'])

    popular_animes = list(model_df.sort_values(
        by='total_ratings', ascending=False)['Name'].values[:30])

    random.shuffle(popular_animes)

    if "popular_animes" not in st.session_state:
        st.session_state.popular_animes = popular_animes

    # Initialize session state
    if "current_anime" not in st.session_state:
        st.session_state.current_anime = 0

    if "final_anime" not in st.session_state:
        st.session_state.final_anime = []

    if "rating" not in st.session_state:
        st.session_state.rating = []

    if st.session_state.current_anime < len(st.session_state.popular_animes) and len(st.session_state.rating) <= 4:
        anime_name = st.session_state.popular_animes[st.session_state.current_anime]
        col1, col2 = st.columns([1, 3])
        with col1:
            image_name = anime_name
            image_file = "_".join(image_name.split(" ")) + ".jpg"
            image_path = os.path.join(Thumbnail_folder, image_file)
            image = Image.open(image_path)
            st.write("***{}***".format(image_name))
            st.image(image, use_column_width=True)

        with col2:
            st.write("#")
            row = model_df[model_df["Name"] == image_name]
            write_anime_details_wide(row, image_name)

        st.subheader("Rate {}:".format(anime_name))

        col1, col2 = st.columns([2, 1])

        with col1:

            answer = st.slider("Your rating", 1, 10)

        with col2:
            col21, col22, col33 = st.columns([0.75, 2, 1.5])

            with col21:
                st.write("#")

                if st.button("Next") and answer:
                    st.session_state.rating.append(answer)
                    st.session_state.final_anime.append(anime_name)
                    st.session_state.current_anime += 1

                    st.experimental_rerun()

            with col22:
                st.write("#")
                if st.button("Did not watch this anime"):
                    st.session_state.current_anime += 1

                    # Clear the input field
                    st.experimental_rerun()

            with col33:
                st.write("#")
                if st.button("Reset"):
                    st.session_state.current_anime = 0
                    st.session_state.rating = []
                    st.experimental_rerun()

    else:

        st.subheader("Here are your personalized recommendations!!")

        st.write("#")

        final_anime = st.session_state.final_anime
        rating = st.session_state.rating

        recommendation_list = get_weighted_recommendations(final_anime, rating,
                                                           cosine_sim=cosine_sim, num_recommendations=10, indices=indices, model_df=model_df)

        cols = st.columns(5)

        for j, col in enumerate(cols):
            with col:
                image_name = recommendation_list[j]
                image_file = "_".join(image_name.split(" ")) + ".jpg"
                image_path = os.path.join(Thumbnail_folder, image_file)
                image = Image.open(image_path)
                st.write("***{}***".format(image_name))
                with st.expander("Know more"):
                    row = front_end_data[front_end_data["Name"] == image_name]
                    write_anime_details(row, image_name)

                st.image(image, use_column_width=True)

    st.write("#")
    st.divider()

    st.header("🕵️‍♂️Anime Whisperer🗣️: Speak the Plot📖, Discover the Title🔍!")

    st.write("#")

    # Initialize the session state
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ''

    if 'closest_anime' not in st.session_state:
        st.session_state.closest_anime = pd.Series()

    if 'anime_index' not in st.session_state:
        st.session_state.anime_index = 0

    if 'start_over_clicked' not in st.session_state:
        st.session_state.start_over_clicked = False

    if st.session_state.closest_anime.empty or st.session_state.start_over_clicked:
        st.session_state.user_input = st.text_area(
            "Enter the Anime story")
        if st.button('Submit'):
            st.session_state.closest_anime = pd.Series(
                get_anime_name_based_on_story(st.session_state.user_input, tfidf=tfidf, tfidf_matrix=tfidf_matrix, model_df=model_df))
            st.session_state.start_over_clicked = False

    else:
        anime_name = st.session_state.closest_anime[st.session_state.anime_index]

        col1, col2 = st.columns([1, 3])
        with col1:
            image_name = anime_name
            image_file = "_".join(image_name.split(" ")) + ".jpg"
            image_path = os.path.join(Thumbnail_folder, image_file)
            image = Image.open(image_path)
            st.subheader(image_name)
            st.image(image, use_column_width=True)

        with col2:
            st.write("#")
            row = model_df[model_df["Name"] == anime_name]
            write_anime_details_wide(row, image_name)

        st.markdown(f"The Anime you're talking about is: **{anime_name}**")
        st.write(
            "Is this not the Anime you're Searching for?, Click Next to catch the Anime you're looking for.")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button('Previous Anime'):
                if st.session_state.anime_index > 0:
                    st.session_state.anime_index -= 1

        with col2:
            if st.button('Start Over'):
                st.session_state.start_over_clicked = True
            st.write("Want to share a different story?")

        with col3:
            if st.button('Next Anime'):
                if st.session_state.anime_index < len(st.session_state.closest_anime) - 1:
                    st.session_state.anime_index += 1

if selected == "Dashboard":

    import plotly.io as pio
    import plotly.express as px
    from scipy.stats import gaussian_kde
    import plotly.graph_objects as go
    import re
    import textwrap
    from plotly.subplots import make_subplots

    pio.templates.default = "simple_white"

    def extract_year(text):
        match = re.search(r'\d{4}', text)
        if match:
            return match.group()
        return np.nan

    def human_format(num):
        magnitude = 0
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0
        return '%.2f%s' % (num, ['', 'K', 'M', 'B', 'T', 'P'][magnitude])

    def convert_to_minutes(time_string):
        time_string = time_string.split('per')[0]
        time_string = time_string.strip()
        if time_string == 'Unknown':
            return time_string
        time_split = time_string.split('.')
        total_minutes = 0
        for t in time_split:
            t = t.strip()
            if 'hr' in t:
                total_minutes += int(t.replace('hr', '').strip()) * 60
            elif 'min' in t:
                total_minutes += int(t.replace('min', '').strip())
            elif 'sec' in t:
                total_minutes += int(t.replace('sec', '').strip()) / 60
        return total_minutes

    current_path = os.path.dirname(__file__)

    df = pd.read_csv(os.path.join(
        current_path, "../../Data/Data_webapp.csv"))

    st.write("#")

    st.header(":earth_americas: Anime Landscape: :roller_coaster: A Journey Through Ratings, Genres, and Time :hourglass:")

    st.write("""This dashboard is our joint venture into the intricate world of anime. Together, we'll navigate the nuances of different anime types, uncover how they fare in terms of ratings, and dive into the colorful spectrum of anime genres. Our exploration will also take us through the evolution of anime across time, tracing both its growth in volume and progress in quality. So, ready to enhance your anime insight and appreciation? Let's embark on this fascinating journey!""")

    st.divider()

    st.header("Anime Types :tv: :movie_camera: :musical_note:")

    st.write("\n")

    st.write("""Anime is a diverse medium with a variety of different types and formats. Here's what each of the categories you listed typically means:

***TV:*** These are standard series airing on TV, with episode counts varying widely. Think "Naruto" or "Attack on Titan".

***Movie:*** Standalone films or movies linked to existing series. Notable examples include "Spirited Away" and "Your Name".

***OVA*** (Original Video Animation): Anime released directly to home video, often offering unique, highproduction quality content.

***Special:*** Extra episodes or movies providing additional content or alternative plot lines, not part of the main storyline.

***ONA*** (Original Net Animation): Internet-distributed anime, a format becoming more popular with the rise of online streaming.

***Music:*** Animated music videos or short animations set to music, either standalone or linked to a series.""")

    fig = px.histogram(df, x="Type", color="Type",
                       nbins=len(df['Type'].unique()),
                       category_orders={
                           'Type': df['Type'].value_counts().index.tolist()}
                       )

    fig.update_layout(
        xaxis_title_text='Type',
        yaxis_title_text='Count',
        bargap=0.1,
        barmode='relative'
    )

    st.write("#")
    st.subheader("Distribution of Anime Types")
    st.plotly_chart(fig, use_container_width=True)

    st.write("""Let us consolidate 'TV', 'OVA' (Original Video Animation), and 'ONA' (Original Net Animation) under the single banner of 'TV'. As These categories share common traits — they're episodic, often have continuous narratives, and are geared towards regular viewing. While there are minor differences, such as production quality or target audience, they are largely similar when compared with formats like 'Movies' or 'Specials'.

Meanwhile, the other types - 'Movies', 'Specials', and 'Music' - retain their individual classifications. These formats have distinct characteristics that set them apart. 'Movies' usually represent standalone experiences or extensions of an existing series but with higher production values. 'Specials' often supplement a series with additional or side-story content. 'Music' represents music videos or short animations accompanying a song, presenting a unique form of anime content. These distinctive aspects make it important to analyze these types separately.

By implementing this grouping, it simplifies our dataset, enabling us to more efficiently pinpoint broader trends between the main methods of anime distribution. This approach ensures we do not miss out on underlying patterns while still providing a comprehensive, easy-to-understand analysis""")
    st.write("#")

    col_1, col_2 = st.columns([3, 7])

    with col_1:

        ratings = df['Score']

        trace1 = go.Histogram(
            x=ratings,
            histnorm='probability density',
            opacity=0.75,
            name='Histogram',
            nbinsx=50,
            marker_color='#1f77b4',
        )

        density = gaussian_kde(ratings)
        xs = np.linspace(min(ratings), max(ratings), 200)
        ys = density(xs)

        trace2 = go.Scatter(
            x=xs,
            y=ys,
            mode='lines',
            name='KDE',
            line=dict(color='#FF2F0E'),
        )

        data = [trace1, trace2]

        layout = go.Layout(
            xaxis=dict(title='Rating', range=[3, 9.5]),
            yaxis=dict(title='Density',  range=[
                       0, max(ys)], automargin=True),
            bargap=0.2,
            bargroupgap=0.1,
            height=600,
            title={
                'text': "Distribution of Anime Ratings",
                'y': 1.0,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'}
        )

        fig = go.Figure(data=data, layout=layout)
        fig.update_layout(showlegend=False)

        st.plotly_chart(fig, use_container_width=True)

    with col_2:
        anime_type_mapping = {"TV": "TV", "Movie": "Movie", "OVA": "TV",
                              "Special": "Special", "ONA": "TV", "Music": "Music"}
        df['Type'] = df['Type'].map(anime_type_mapping)

        fig = px.box(df, x="Type", y="Score", color="Type")
        fig.update_layout(
            xaxis_title_text='Type',
            yaxis_title_text='Rating',
            height=600,
            title={
                'text': 'Distribution of Rating by Type of Anime',
                'y': 1.0,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'}
        )

        st.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(df, x="Score", color="Type", nbins=100, opacity=0.6)

    fig.update_layout(
        xaxis_title_text='Rating',
        yaxis_title_text='Count',
        barmode='overlay',
        bargap=0.1,
        bargroupgap=0.2,
        height=400,
        title={
            'text': "Distribution of Rating based on Anime Type",
            'y': 1.0,
            'x': 0.55,
            'xanchor': 'center',
            'yanchor': 'top'},
        legend=dict(
            x=0,
            y=1,
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=12,
                color="black"
            )
        )
    )

    st.write("#")

    st.plotly_chart(fig, use_container_width=True)

    st.write("""With majority of ratings falling between the 5 to 8 stars mark. This indicates a strong central tendency, where most values tend to cluster around the central or 'average' value. This is not uncommon in the world of entertainment where most productions are rated averagely, with few outliers achieving either extremely high or extremely low ratings.

When we dig deeper into these ratings based on the type of anime, the patterns become even more interesting. Despite the differences in the anime types - TV, Movie, OVA, Special, and ONA - all of them follow a similar distribution and range of ratings. This suggests that irrespective of the type, anime content generally maintains a consistent level of quality that resonates with viewers, resulting in a relatively standard rating distribution.

However, anime type Music stands out from this observation. While it maintains a similar overall distribution, its average rating is noticeably lower. This might indicate a divergence in viewer preference when it comes to Musical anime or potentially highlight a niche audience that appreciates this type of anime, contributing to a lower general rating. What is also interesting to note is the longer right tail in the distribution for Musical anime, signifying a greater spread of higher ratings. This might suggest a subset of viewers who rate these productions very highly, again hinting at a smaller fanbase.""")

    st.divider()

    new_df = df.copy(deep=True)
    new_df = new_df[new_df['Episodes'] != 'Unknown']
    new_df = new_df[new_df['Duration'] != 'Unknown']
    new_df['Episodes'] = new_df['Episodes'].astype(int)
    new_df['Duration'] = new_df['Duration'].apply(convert_to_minutes)
    new_df['Total_Duration'] = new_df['Duration'] * new_df['Episodes']

    st.header("How long does it take to watch an anime? :clock8:")

    kde_dict = {}
    range_dict = {}
    for t in new_df['Type'].unique():
        df_type = new_df[new_df['Type'] == t]
        df_type['Total_Duration'] = pd.to_numeric(
            df_type['Total_Duration'], errors='coerce') / 60
        kde = gaussian_kde(df_type['Total_Duration'].dropna())
        kde_dict[t] = kde
        range_dict[t] = (df_type['Total_Duration'].min(), min(
            df_type['Total_Duration'].max(), 1500/60))

    all_types = list(kde_dict.keys())

    selected_types = st.multiselect(
        'Select types', all_types, default=['TV'])
    st.write("#")

    traces = []
    x_range = []
    for t in selected_types:
        x = np.linspace(*range_dict[t], 1000)
        y = kde_dict[t](x)
        traces.append(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            name=t
        ))
        x_range.append(range_dict[t])

    x_range = (min([r[0] for r in x_range]), max([r[1] for r in x_range]))

    fig = go.Figure(data=traces)
    fig.update_layout(xaxis_title='Total Duration (Hours)', xaxis_range=x_range, yaxis_title='Density', title={
        'text': "Distribution of Anime Run Time by Type",
        'y': 1.0,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'}
    )

    st.plotly_chart(fig, use_container_width=True)

    st.write("""The duration of anime presents an intriguing dimension of variability, influenced largely by the type of the anime. A clear testament to this diversity is TV anime, which typically boasts the longest runtimes. This makes perfect sense as TV series have the luxury of time to delve deep into character arcs, unravel subplots, and span across multiple episodes or even seasons.

What's particularly captivating is the dual nature of Movie-type anime, exhibiting bimodal runtime distribution with peaks at around 10 minutes and 1.5 hours. This fascinating observation could imply two distinct types of productions: shorter pieces that might serve as exciting trailers or introductions to new TV series, and feature-length movies.""")

    st.divider()
    st.header("Anime Genres :sparkles:")

    splitted_genres = df['Genres'].str.split(', ', expand=True)
    stacked_genres = splitted_genres.stack()
    genre_counts = pd.DataFrame({"Genre": stacked_genres.value_counts(
    ).index, "Count": stacked_genres.value_counts().values})
    genre_counts['Percentage'] = np.round(genre_counts['Count'].apply(
        lambda count: (count/genre_counts['Count'].sum())*100), 2)
    genre_counts = genre_counts[genre_counts['Genre'] != 'Unknown']

    fig = px.bar(genre_counts, x="Genre", y="Count",
                 color="Genre", hover_data=['Percentage'])
    fig.update_xaxes(tickangle=270)
    fig.update_layout(showlegend=False, xaxis_title_text='Genre', yaxis_title_text='Count', height=600, title={
        'text': "Popular Genre for Anime",
        'y': 1.0,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

    st.write("#")
    st.plotly_chart(fig, use_container_width=True)

    genre_order = genre_counts['Genre'].tolist()
    new_df = df.copy(deep=True)
    new_df = new_df[new_df['Genres'] != 'Unknown']
    new_df['Genres'] = new_df['Genres'].apply(
        lambda x: x.split(', '))  # Convert genre strings to lists
    exploded_df = new_df.explode('Genres')

    fig = px.box(exploded_df, x='Genres', y='Score', color='Genres',
                 category_orders={'Genres': genre_order})
    fig.update_xaxes(tickangle=270)
    fig.update_layout(showlegend=False, xaxis_title_text='Genre', yaxis_title_text='Rating', title={
        'text': "Distribution of Rating by Genre",
        'y': 1.0,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

    st.write("#")
    st.plotly_chart(fig, use_container_width=True)

    st.write("""Upon reviewing the distribution of genres in our anime dataset, a clear preference for certain genres emerges among viewers. Comedy is the reigning champion, standing as the most frequently occurring genre. It underscores the universal appeal of humor and the joy audiences find in light-hearted, amusing narratives.

Following Comedy, the popularity list continues with Action, indicating a strong appetite for exhilarating stories full of conflict, physical feats, and heroic exploits. Fantasy, offering a delightful escape from reality through mystical and imaginative elements, claims the third spot. Adventure and Sci-Fi, genres full of exploration and forward-thinking concepts, respectively, also enjoy significant popularity. The sixth most frequent genre, Drama, showcases viewers' appreciation for emotionally-charged storylines, profound character growth, and poignant life situations.

Overall, these findings offer a fascinating glimpse into the world of anime, revealing what genre elements most captivate viewers' imaginations and maintain their engagement.""")

    st.divider()

    st.header("Anime over the years :calendar:")

    st.write("#")

    # Assuming the column with the dates is named 'Date'
    df['Aired_Year'] = df['Aired'].apply(extract_year)
    new_df = df.dropna(subset=['Aired_Year'])

    # Convert the extracted Year to numeric
    new_df['Aired_Year'] = new_df['Aired_Year'].astype(int)

    vc = new_df['Aired_Year'].value_counts().sort_index()
    anime_year = pd.DataFrame({"Aired_Year": vc.index, "Count": vc.values})

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=anime_year['Aired_Year'],
            y=anime_year['Count'],
            mode='lines'
        )
    )

    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Number of Anime', title={
            'text': "Number of Anime Released over Years",
            'y': 1.0,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'}
    )

    st.write("#")
    st.plotly_chart(fig, use_container_width=True)
    st.write("""The anime industry has seen remarkable growth since its early stages. From a humble beginning of releasing one to two productions per year, the industry expanded significantly. The ***1960s*** marked the ***initial growth*** phase, with an average of ***five to six releases per year***, showcasing an increasing interest in anime.

A dramatic surge in anime production occurred, ***peaking in 2016*** with a remarkable ***638 titles released in a single year***. This era represents the peak popularity of anime worldwide.

The post-2016 decline could signal market saturation, or perhaps a shift from quantity to quality, impacted by economic or technological factors. Overall, this graph presents a compelling view of the anime industry's evolution over the decades, underlining its growth, peaks, and subsequent adjustments.""")
    st.write("#")

    temp_df = new_df.copy(deep=True)
    temp_df['Genres'] = temp_df['Genres'].str.split(',')

    df_exploded = temp_df.explode('Genres')

    pivot_df = df_exploded.groupby(['Aired_Year', 'Genres']).count().reset_index(
    ).pivot(index='Aired_Year', columns='Genres', values='Name').fillna(0)

    all_genres = pivot_df.columns.tolist()

    st.subheader('Anime Released by genre over years')

    selected_genres = st.multiselect(
        'Select genres', all_genres, default=['Adventure', 'Fantasy'])

    traces = []
    for genre in selected_genres:
        traces.append(go.Scatter(
            x=pivot_df.index,
            y=pivot_df[genre],
            mode='lines',
            name=genre
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(xaxis_title='Year', yaxis_title='# Anime', legend=dict(
        x=0,
        y=1,
        traceorder="normal",
        font=dict(
            family="sans-serif",
            size=12,
            color="black"
        )))

    st.plotly_chart(fig, use_container_width=True)

    df_grouped = np.round(temp_df.groupby('Aired_Year')['Score'].mean(), 2)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_grouped.index,
        y=df_grouped.values,
        mode='lines',
        name='Average Rating',
        line=dict(color="#4E79A7")
    ))

    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Average Rating', title={
            'text': "Average Rating over the years",
            'y': 1.0,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'}
    )

    st.plotly_chart(fig, use_container_width=True)

    st.write("""This clearly highlights a simultaneous increase in both quantity and quality in the anime industry over the decades. It traces the average rating of anime from the ***modest score of 5 in the 1910s*** to a significant ***6.5 by the 2010s***. This upward trajectory showcases the industry's consistent efforts in producing better-quality content, aligning with its exponential growth in production numbers.""")

    st.divider()

    st.header("Top 10 Anime :trophy:")

    top_movies = temp_df[['Name', 'Score']].sort_values(
        by='Score', ascending=False).head(10)

    top_movies['Name'] = ['<br>'.join(textwrap.wrap(
        i, 15, break_long_words=False)) for i in top_movies['Name']]

    fig = px.bar(top_movies, x='Name', y='Score', color='Name')

    fig.update_layout(showlegend=False, xaxis_title_text='Anime', yaxis_title_text='Rating', title={
        'text': "Top 10 Anime by Rating",
        'y': 1.0,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

    for xi, yi in zip(top_movies['Name'], top_movies['Score']):
        fig.add_annotation(x=xi, y=yi,
                           text=str(yi),
                           showarrow=False,
                           font=dict(
                               size=12,
                               color='Black'
                           ),
                           align='center',
                           ax=0,
                           ay=-yi,
                           yshift=10
                           )

    fig.update_xaxes(tickangle=0, tickfont=dict(size=12))

    st.write("#")
    st.plotly_chart(fig, use_container_width=True)

    top_movies = temp_df[['Name', 'Members']].sort_values(
        by='Members', ascending=False).head(10)

    top_movies['Name'] = ['<br>'.join(textwrap.wrap(
        i, 15, break_long_words=False)) for i in top_movies['Name']]

    fig = px.bar(top_movies, x='Name', y='Members', color='Name')

    fig.update_layout(showlegend=False, xaxis_title_text='Anime', yaxis_title_text='Members', title={
        'text': "Top 10 Anime with biggest community",
        'y': 1.0,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

    for xi, yi in zip(top_movies['Name'], top_movies['Members']):
        fig.add_annotation(x=xi, y=yi,
                           text=human_format(yi),
                           showarrow=False,
                           font=dict(
                               size=12,
                               color='Black'
                           ),
                           align='center',
                           ax=0,
                           ay=-yi,
                           yshift=10
                           )

    fig.update_xaxes(tickangle=0, tickfont=dict(size=12))
    st.write("#")
    st.plotly_chart(fig, use_container_width=True)

    st.write("""The world of anime offers an exciting range of choices for viewers, each show boasting its unique charm and appeal. According to the ratings, "Fullmetal Alchemist Brotherhood" is the top choice, scoring an impressive 9.19, hinting at a mix of engaging storylines, well-crafted characters, and possibly, top-notch animation. Other highly-rated series such as "Attack on Titan Final Season", "Steins;Gate", and "Hunter x Hunter" seem to have also won the hearts of viewers, showing their strength in storytelling.

But it gets intriguing when we consider the size of anime communities. Even though "Fullmetal Alchemist Brotherhood" has the highest rating, it's actually "Death Note" and "Attack on Titan" that have the most members. And it's interesting to note that some series with the biggest fandoms, like "Sword Art Online" and "One Punch Man", don't even make it to the top-rated list. This suggests that a high rating isn't the only thing that matters to viewers. It could be that certain themes or styles resonate more with a larger audience, or perhaps the series was more accessible or well-marketed. It goes to show that anime is truly a diverse medium with something for everyone!""")

    st.divider()

    st.header("MPAA Ratings :movie_camera:")

    ratings_score_df = get_ratings_score(temp_df)

    color_dict = {'PG-13 - Teens 13 or older': '#1f77b4', 'G - All Ages': '#ff7f0e', 'Rx - Hentai': '#2ca02c',
                  'R - 17+ (violence & profanity)': '#d62728', 'R+ - Mild Nudity': '#9467bd', 'PG - Children': '#8c564b'}

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=ratings_score_df['Rating'], y=ratings_score_df['Count'],
                  name="Number of Anime",   marker_color=ratings_score_df['Rating'].map(color_dict)), secondary_y=False)
    fig.add_trace(go.Scatter(x=ratings_score_df['Rating'], y=ratings_score_df['Average_score'],
                  name="Average Score", mode='lines+markers', line=dict(color='rgba(0, 0, 0, 0.5)')), secondary_y=True)
    fig.update_xaxes(title_text="Rating")
    fig.update_yaxes(title_text="Number of Anime", secondary_y=False)
    fig.update_yaxes(title_text="Average Score", secondary_y=True)
    fig.update_layout(showlegend=False, xaxis_title_text='MPAA Film Ratings', yaxis_title_text='Number of Anime', title={
        'text': "Number of Anime and Average Score by MPAA Film Ratings",
        'y': 1.0,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

    st.write("#")
    st.plotly_chart(fig, use_container_width=True)

    st.write("""The majority of anime shows are rated 'PG-13 - Teens 13 or older', reflecting their wide appeal, but an interesting twist emerges with the 'R - 17+ (violence & profanity)' category, which despite having fewer shows, boasts the highest Average score of 7.1. Conversely, the 'G - All Ages' category, though catering to a broad audience, records a lower Average score of 6.03, pointing towards a potential disconnect between viewership size and content appreciation. This highlights the diverse interests and critical tastes within the global anime community""")

    st.divider()

    st.header(":sparkles: Conclusion and Reflections :sparkles:")

    st.write("""In conclusion, our journey through the world of anime highlights its diversity, adaptability, and the compelling storytelling that resonates with a global audience. Despite shifts in production volume, the industry's focus on delivering quality content has remained steadfast. As we reflect on our findings, we better understand and appreciate this rich medium and its impact on popular culture. Moving forward, let's anticipate continued innovation and captivating narratives from this dynamic industry. Thank you for joining me on this enlightening exploration into anime!""")
