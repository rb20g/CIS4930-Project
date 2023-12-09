# Data processing
import pandas as pd
import numpy as np
import scipy.stats

# Visualization
import seaborn as sns
from matplotlib import pyplot as plt

# Similarity
from sklearn.metrics.pairwise import cosine_similarity




"""
------------------------------------------------
 merged.csv info:
------------------------------------------------
 0   user_id          100836 non-null  int64  
 1   movie_id         100836 non-null  int64  
 2   rating           100836 non-null  float64
 3   title            100836 non-null  object 
 4   genres           100836 non-null  object 
 5   is_adult         100762 non-null  float64
 6   runtime_min      100761 non-null  float64
 7   avgRating_IMDB   100762 non-null  float64
 8   numVotes_IMDB    100762 non-null  float64
 9   avgRating_users  0 non-null       float64
 10  numVotes_users   0 non-null       float64
 11  timestamp        100836 non-null  int64  
 12  IMDB_id          100836 non-null  int64  
------------------------------------------------
"""

# Function to user-movie rating data from a CSV file
def load_data(filepath):
    ratings = pd.read_csv(filepath)
    print(ratings.head(), "\n")
    print('The ratings dataset has', ratings['user_id'].nunique(), 'unique users', "\n")
    print('The ratings dataset has', ratings['movie_id'].nunique(), 'unique movies', "\n")
    print('The ratings dataset has', ratings['rating'].nunique(), 'unique ratings', "\n")
    print('The unique ratings are', sorted(ratings['rating'].unique()), "\n")
    return ratings    

# Function to preprocess ratings data
def preprocess_ratings(ratings):
    # Aggregate ratings by movie title, calculate mean rating and number of ratings
    agg_ratings = ratings.groupby('title').agg(mean_rating=('rating', 'mean'),
                                      number_of_ratings=('rating', 'count')).reset_index()
    
    # Filter movies with more than 100 ratings
    agg_ratings_GT100 = agg_ratings[agg_ratings['number_of_ratings'] > 100]

    # Visualize the aggregated ratings
    #visualize_ratings(agg_ratings_GT100)

    # Merge ratings with movies that have more than 100 ratings, get rid of those with less than 100 ratings
    df_GT100 = pd.merge(ratings, agg_ratings_GT100[['title']], on='title', how='inner')
   
    return df_GT100
    
# Function to visualize ratings data
def visualize_ratings(agg_ratings_GT100):
    sns.jointplot(x='mean_rating', y='number_of_ratings', data=agg_ratings_GT100)
    plt.show()    

# Function to update the aggregated ratings data
def update_agg_ratings(agg_ratings_GT100):
    # Group by 'title' and calculate mean rating and number of ratings
    movie_stats = agg_ratings_GT100.groupby('title').agg({'rating': ['mean', 'count']})

    # Reset the index to make 'title' a column again
    movie_stats = movie_stats.reset_index()

    # Flatten the MultiIndex in columns
    movie_stats.columns = ['_'.join(col).strip() for col in movie_stats.columns.values]

    # Rename the columns
    movie_stats.rename(columns={'title_': 'title', 'rating_mean': 'mean_rating', 'rating_count': 'num_ratings'}, inplace=True)

    # Create a dictionary mapping column names in movie_stats to column names in agg_ratings_GT100
    column_mapping = {'mean_rating': 'avgRating_users', 'num_ratings': 'numVotes_users'}

    # Loop over the column mappings
    for src_col, dst_col in column_mapping.items():
        # Create a dictionary mapping movie titles to the values in src_col
        value_dict = movie_stats.set_index('title')[src_col].to_dict()
        
        # Update the values in dst_col in agg_ratings_GT100 using the dictionary
        agg_ratings_GT100[dst_col] = agg_ratings_GT100['title'].map(value_dict)

    return agg_ratings_GT100

# Function to create the user-item matrix
def create_user_item_matrix(df_GT100):
    matrix = df_GT100.pivot_table(index='title', columns='user_id', values='rating')
    # print(matrix.head(), "\n")
    # print(matrix[2])  
    # print(matrix.loc['Inception (2010)'])


    matrix_norm = matrix.subtract(matrix.mean(axis=1), axis=0)
    #print(matrix_norm.loc['Inception (2010)'])
    return matrix    

# Function to calculate item similarity matrix
def calculate_item_similarity(matrix_norm):
    item_similarity = matrix_norm.T.corr()
    return item_similarity

# Function to predict a rating for a given user and movie
def predict_rating(matrix_norm, item_similarity, picked_userid, picked_movie):
    # Check if picked_movie is in item_similarity
    if picked_movie not in item_similarity.columns:
        with open('predict_rating_out.txt', 'a') as f:
            f.write(f'\n***********{picked_movie} is not in the item_similarity matrix***********\n\n')
        return None
    #print(matrix_norm[picked_userid])
    picked_userid_watched = pd.DataFrame(matrix_norm[picked_userid].dropna(axis=0, how='all') \
                                    .sort_values(ascending=False)) \
                                    .reset_index() 
    picked_userid_watched.columns = ['title', 'rating']
    
    picked_movie_similarity_score = item_similarity[[picked_movie]].reset_index() \
        .rename(columns={picked_movie:'similarity_score'})
    
    picked_userid_watched_similarity = pd.merge(left=picked_userid_watched,
                                            right=picked_movie_similarity_score,
                                            on='title',
                                            how='inner') \
                                       .sort_values('similarity_score', ascending=False)[:5]
    #print(picked_userid_watched_similarity.head())    
    predicted_rating = round(np.average(picked_userid_watched_similarity['rating'],
                                    weights=picked_userid_watched_similarity['similarity_score']), 6)
    return predicted_rating

# Function to perform item-based recommendation
def item_based_rec(matrix_norm, item_similarity, picked_userid, number_of_similar_items, number_of_recommendations):
    picked_userid_unwatched = pd.DataFrame(matrix_norm[picked_userid].isna()).reset_index()
    picked_userid_unwatched = picked_userid_unwatched[picked_userid_unwatched[1] == True]['title'].values.tolist()
    picked_userid_watched = pd.DataFrame(matrix_norm[picked_userid].dropna(axis=0, how='all')
                                         .sort_values(ascending=False)).reset_index().rename(columns={1: 'rating'})
    
    rating_prediction = {}
    
    for picked_movie in picked_userid_unwatched:
        picked_movie_similarity_score = item_similarity[[picked_movie]]\
            .reset_index().rename(columns={picked_movie:'similarity_score'})
        picked_userid_watched_similarity = pd.merge(left=picked_userid_watched,
                                                    right=picked_movie_similarity_score,
                                                    on='title', how='inner')\
                                                .sort_values('similarity_score', ascending=False)[:number_of_similar_items]
        predicted_rating = round(np.average(picked_userid_watched_similarity['rating'],
                                            weights=picked_userid_watched_similarity['similarity_score']), 6)
        rating_prediction[picked_movie] = predicted_rating
    return sorted(rating_prediction.items(), key=lambda x: x[1], reverse=True)[:number_of_recommendations]

# Main function
def main():
    # Usage example:
    filepath = 'merged.csv'

    # Load data
    ratings = load_data(filepath)
    print("Head of ratings DataFrame:")
    print(ratings.head(), "\n")

    # Preprocess ratings data by aggregating ratings by movie title, 
    # calculate mean rating and number of ratings, and filter out movies with less than 100 ratings
    agg_ratings_GT100 = preprocess_ratings(ratings)
    print("Head of agg_ratings_GT100 DataFrame:")
    print(agg_ratings_GT100.head(), "\n")

    
    agg_ratings_GT100 = update_agg_ratings(agg_ratings_GT100)

    print(agg_ratings_GT100.head(), "\n")



    # Print all rows for user_id = 2
    #print(agg_ratings_GT100[agg_ratings_GT100['user_id'] == 2], "\n")

    # Create a test set by taking one row for each user_id, only if the user has an entry in their 'rating' column
    test = agg_ratings_GT100[agg_ratings_GT100['rating'].notna()].groupby('user_id').apply(lambda x: x.sample(1)).reset_index(drop=True)
    # print rows in test set with user_id = 1
    #print(test[test['user_id'] == 2], "\n")

    # Create a training set by removing the test set rows from the matrix
    train = agg_ratings_GT100.drop(test.index)

    print("Head of train DataFrame:")
    print(train.head(), "\n")
    print("Head of test DataFrame:")
    print(test.head(), "\n")

    # Write the train and test sets to CSV files
    train.to_csv('train.csv')
    test.to_csv('test.csv')

    # Print all rows for user_id = 2
    #print(train[train['user_id'] == 2], "\n")
    #print(test[test['user_id'] == 2], "\n")


    
    # Create user-item matrix and normalize the matrix by subtracting the mean rating of each movie from the ratings
    matrix_norm = create_user_item_matrix(train)
    print("Head of matrix_norm DataFrame:")
    print(matrix_norm.head(), "\n")


    item_similarity = calculate_item_similarity(matrix_norm)
    print("Head of item_similarity DataFrame:")
    print(item_similarity.head(), "\n")

    actual_rating_list = []
    predicted_rating_list = []
    picked_userid_list = []
    actual_minus_predict_list = []
    with open('predict_rating_out.txt', 'w') as f:
        pass

    # Predict rating for a all users and movie in the test set
    # Loop through all users in the test set
    for picked_userid in test['user_id'].unique():
        # Get the movie title from the test set
        picked_movie = test[test['user_id'] == picked_userid]['title'].values[0]
        # Get actual rating from the test set
        actual_rating = test[test['user_id'] == picked_userid]['rating'].values[0]
        actual_rating_list.append(actual_rating)
        # Calculate the predicted rating
        predicted_rating = predict_rating(matrix_norm, item_similarity, picked_userid, picked_movie)
        predicted_rating_list.append(predicted_rating)
        actual_minus_predict_list.append(abs(predicted_rating - actual_rating)) 
        
        # Append to the file
        with open('predict_rating_out.txt', 'a') as f:
            f.write(f'The predicted rating for {picked_movie} by user {picked_userid} is {predicted_rating}\n')
            f.write(f'The actual rating for {picked_movie} by user {picked_userid} is {actual_rating}\n')
            f.write("------------------------------------------------------------------\n")
         
    rating_points = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    data = pd.DataFrame({'UserID': picked_userid_list, 'Predicted Ratings': predicted_rating_list, 'Actual Ratings': actual_rating_list, 'Predicted - Actual Rating': actual_minus_predict_list})
    sns.set(style="whitegrid")
    plt.figure(figsize=(16, 8))
    sns.scatterplot(x='UserID', y='Predicted - Actual Rating', data=data, label='Predicted Ratings')
    #sns.scatterplot(x='UserID', y='Actual Ratings', data=data, label='Actual Ratings')
    plt.xlabel('UserID')
    plt.ylabel('Rating')
    plt.title('Difference of Predicted and Actual User Ratings')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # # Perform item-based recommendation for a specific user
    # recommended_movies = item_based_rec(matrix_norm, item_similarity, picked_userid=1, number_of_similar_items=5, number_of_recommendations=3)
    # print(f'The top 3 recommended movies for user 1 are {recommended_movies}\n')
    # print("Done\n")


if __name__ == "__main__":
    main()
