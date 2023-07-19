import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
import re
import plotly.graph_objs as go


anime = pd.read_csv('Data/anime.csv')
temp_df = anime.loc[anime['Score'] != "Unknown"]
temp_df['Score'] = temp_df['Score'].astype(float)


# Split the genres column into multiple genres
splitted_genres = temp_df['Genres'].str.split(', ', expand=True)

# Stack all the genres into a single column
stacked_genres = splitted_genres.stack()

# Count the occurrences of each genre
genre_counts = pd.DataFrame({"Genre": stacked_genres.value_counts(
).index, "Count": stacked_genres.value_counts().values})

genre_counts['Percentage'] = np.round(genre_counts['Count'].apply(
    lambda count: (count/genre_counts['Count'].sum())*100), 2)


def extract_year(text):
    match = re.search(r'\d{4}', text)
    if match:
        return match.group()
    return np.nan


# Assuming the column with the dates is named 'Date'
temp_df['Aired_Year'] = temp_df['Aired'].apply(extract_year)
temp_df.dropna(subset=['Aired_Year'], inplace=True)

# Convert the extracted Year to numeric
temp_df['Aired_Year'] = temp_df['Aired_Year'].astype(int)


# Here we assume your data is in a dataframe called 'df'

# Split the genres by comma into a list (replace ',' with your actual separator if different)
new_df = temp_df.copy(deep=True)
new_df['Genres'] = new_df['Genres'].str.split(',')

# Explode the dataframe on the 'Genres' column
df_exploded = new_df.explode('Genres')

# Now groupby 'Year' and 'Genres', count the movies, reset index, and pivot the dataframe
pivot_df = df_exploded.groupby(['Aired_Year', 'Genres']).count().reset_index(
).pivot(index='Aired_Year', columns='Genres', values='Name').fillna(0)

# The list of all genres
all_genres = pivot_df.columns.tolist()

st.title("Number of Movies Released per Year by Genre")


def plot():

    # Create a multiselect box for the genres
    selected_genres = st.multiselect(
        'Choose genres', all_genres, default=['Adventure', 'Fantasy'])

    traces = []

    # Create the traces
    for genre in selected_genres:
        traces.append(go.Scatter(
            x=pivot_df.index,
            y=pivot_df[genre],
            mode='lines',
            name=genre
        ))

    # Create the figure
    fig = go.Figure(data=traces, layout=go.Layout(
        title='Number of Movies Released per Year by Genre',
        xaxis={'title': 'Year'},
        yaxis={'title': 'Number of Movies'}))

    return fig


# Display the figure

st.markdown('### Metrics')
col1, col2, col3 = st.columns(3)
col1.metric("Temperature", "70 °F", "1.2 °F")
my_expander = st.expander("See Plot")
with my_expander:
    fig = plot()
    st.plotly_chart(fig)

col2.metric("Wind", "9 mph", "-8%")
col3.metric("Humidity", "86%", "4%")
