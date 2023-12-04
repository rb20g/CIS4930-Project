# Data processing
import pandas as pd
import numpy as np
import scipy.stats

# Visualization
import seaborn as sns
from matplotlib import pyplot as plt

# Similarity
from sklearn.metrics.pairwise import cosine_similarity

# Read in data from the ratings set
ratings = pd.read_csv('ml-25m/ml-25m/ratings.csv')

# Look at the data from the ratings set
ratings.head()

# Get the information from the ratings dataset
# ratings.info()

# Number of users
print('The ratings dataset has', ratings['userId'].nunique(), 'unique users')

# Number of movies
print('The ratings dataset has', ratings['movieId'].nunique(), 'unique movies')

# Number of ratings
print('The ratings dataset has', ratings['rating'].nunique(), 'unique ratings')

# List of unique ratings
print('The unique ratings are', sorted(ratings['rating'].unique()))

# Read in data from movies dataset
movies = pd.read_csv('ml-25m/ml-25m/movies.csv')

# Look at the data
movies.head()

# Merge ratings and movies datasets
df = pd.merge(ratings, movies, on='movieId', how='inner') \
 \
    # Look at the data
df.head()

# Aggregate by movie
agg_ratings = df.groupby('title').agg(mean_rating=('rating', 'mean'),
                                      number_of_ratings=('rating', 'count')).reset_index()

# Keep the movies with over 100 ratings
agg_ratings_GT100 = agg_ratings[agg_ratings['number_of_ratings'] > 100]

# Check the information of the dataframe
# agg_ratings_GT100.info()

# Check popular movies
agg_ratings_GT100.sort_values(by='number_of_ratings', ascending=False).head()

# Joint plot visualization of the correlation between the average rating and the number of ratings of a movie
sns.jointplot(x='mean_rating', y='number_of_ratings', data=agg_ratings_GT100)
# plt.show() //displays the joint plot

# Merge data so we only keep the movies with more than 100 ratings since most from the list have less than 150 ratings
df_GT100 = pd.merge(df, agg_ratings_GT100[['title']], on='title', how='inner')
df_GT100.info()

# Number of users after merge
print('The ratings dataset has', df_GT100['userId'].nunique(), 'unique users')

# Number of movies after merge
print('The ratings dataset has', df_GT100['movieId'].nunique(), 'unique movies')

# Number of ratings after merge
print('The ratings dataset has', df_GT100['rating'].nunique(), 'unique ratings')

# List of unique ratings
print('the unique ratings are', sorted(df_GT100['rating'].unique()))

# creates user-item matrix, where the rows are movies and the columns are users
# value of matrix is the user rating of the movie if there is a rating, if no rating 'NaN'
matrix = df_GT100.pivot_table(index='title', columns='userId', values='rating')
matrix.head()

# Normalize user-item matrix by subtracting the average rating of each movie
# Mean-center cosine similarity found by cosine similarity based on the normalized data
matrix_norm = matrix.subtract(matrix.mean(axis=1), axis=0)
matrix_norm.head()
# After normalization, ratings less than the movie's average rating get a negative value
# And ratings more than the movie's average rating get positive values

# Measuring item similarity matrix using Pearson correlation
item_similarity = matrix_norm.T.corr()
item_similarity.head()

# Next is predicting user's rating for one movie, attempting this on user 1 and the movie American Pie as example
# Step 1. create a list of movies that the user has watched and rated
# Step 2. Rank the similarities between the movies the user has rated, and the movie we are trying to predict
# Step 3. select top n movies with the highest similarity scores
# Step 4. calculate the predicted rating using weighted average of similarity scores and the ratings from the user

# Pick a user ID
picked_userid = 1

# Pick a movie
picked_movie = 'American Pie (1999)'

# Movies that the target user has watched
picked_userid_watched = pd.DataFrame(matrix_norm[picked_userid].dropna(axis=0, how='all') \
                                     .sort_values(ascending=False)) \
    .reset_index() \
    .rename(columns={1: 'rating'})
picked_userid_watched.head()

# Similarity score of the specified movie with all other movies
picked_movie_similarity_score = item_similarity[[picked_movie]].reset_index() \
    .rename(columns={'American Pie (1999)': 'similarity_score'})

# Rank the similarities between the movies the user has rated and the movie selected
n = 5
picked_userid_watched_similarity = pd.merge(left=picked_userid_watched,
                                            right=picked_movie_similarity_score,
                                            on='title',
                                            how='inner') \
                                       .sort_values('similarity_score', ascending=False)[:5]

# Take a look at the users watched movies with the highest similarity
picked_userid_watched_similarity.head()

# Calculate the predicted rating using weighted average of similarity scores and the ratings from the user
predicted_rating = round(np.average(picked_userid_watched_similarity['rating'],
                                    weights=picked_userid_watched_similarity['similarity_score']), 6)

print(f'The predicted rating for {picked_movie} by user {picked_userid} is {predicted_rating}')


# Next will create an item-item movie recommendation system:
# Step 1. create a list of movies that the target user has not watched before
# Step 2. loop through the unwatched movie and create predicted scores for each movie
# Step 3. Rank the predicted score of unwatched movie from high to low
# Step 4. Select the top k movies as the recommendations for the target user
# using number_of_similar_items and number_of_recommendations to get the top movies and ratings for the user

# Item-based recommendation function
def item_based_rec(picked_userid=1, number_of_similar_items=5, number_of_recommendations=3):
    import operator
    # Movies that the target has not watched
    picked_userid_unwatched = pd.DataFrame(matrix_norm[picked_userid].isna()).reset_index()
    picked_userid_unwatched = picked_userid_unwatched[picked_userid_unwatched[1] == True]['title'].values.tolist()

    # Movies the user has watched
    picked_userid_watched = pd.DataFrame(matrix_norm[picked_userid].dropna(axis=0, how='all')
                                         .sort_values(ascending=False)).reset_index().rename(columns={1: 'rating'})

    # Dictionary to save the unwatched movie and predict rating pair
    rating_prediction = {}

    # Loop through unwatched movies
    for picked_movie in picked_userid_unwatched:
        # Calculate the similarity score of the picked movie with other movies
        picked_movie_similarity_score = item_similarity[[picked_movie]]\
            .reset_index().rename(columns={picked_movie:'similarity_score'})
        # Rank the similarities between the picked user watched movie and the picked unwatched movie
        picked_userid_watched_similarity = pd.merge(left=picked_userid_watched,
                                                    right=picked_movie_similarity_score,
                                                    on='title', how='inner')\
                                                .sort_values('similarity_score', ascending=False)[:number_of_similar_items]
        # Calculate the predicted rating using weighted average of similarity scores and the ratings for the user
        predicted_rating = round(np.average(picked_userid_watched_similarity['rating'],
                                            weights=picked_userid_watched_similarity['similarity_score']), 6)
        # Save the predicted rating in the dictionary
        rating_prediction[picked_movie] = predicted_rating
        # Return top recommended movies
    return sorted(rating_prediction.items(), key=operator.itemgetter(1), reverse=True)[:number_of_recommendations]


# Get recommendations
recommended_movie = item_based_rec(picked_userid=1, number_of_similar_items=5, number_of_recommendations=3)
print(recommended_movie)




