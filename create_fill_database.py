import sqlite3
import csv
from tqdm import tqdm
import pandas as pd
import numpy as np
from multiprocessing import Pool
import time
import os

# You can use the functions create_movie_table(), create_ratings_table(), and merge_tables() to create and populate the databases in one concurrent process (kind of slow)
# Or you can use the function update_moviesT_parrallel() to create and populate the databases in parallel processes (faster but not done yet)
# imdb data can be found at https://datasets.imdbws.com/
# ml-latest-small data can be found at https://grouplens.org/datasets/movielens/
# The database is created in the same directory as this file
# required pip installs: tqdm

rating_file = 'ml-latest-small/ratings.csv'
movie_file = 'ml-latest-small/movies.csv'
link_file = 'ml-latest-small/links.csv'

imdb_movie_file = 'IMDB-data-BIG/title.basics.tsv'
imdb_rating_file = 'IMDB-data-BIG/title.ratings.tsv'

database_name = 'movie_recommender.db'


# Merge movie table with IMDB data
def add_IMDB_data(file_name):
    with sqlite3.connect(database_name) as conn:
        cursor = conn.cursor()
        if file_name == imdb_rating_file:
            with open(file_name, 'r', encoding='utf-8') as tsvfile:
                reader = csv.reader(tsvfile, delimiter='\t')
                next(reader)
                for row in tqdm(reader):
                    # Check if the 'Imdb_id' variable, with the first 2 letters removed and the rest converted to int, exists in the 'movies' table 
                    Imdb_id = int(row[0][2:])
                    cursor.execute("SELECT 1 FROM movies WHERE IMDB_id=?", (Imdb_id,))
                    data = cursor.fetchone()
                    if data is not None:
                        # If the 'Imdb_id' variable exists in the 'movies' table, update the 'avgRating_IMDB' and 'numVotes_IMDB' variables
                        avgRating = None if row[1] == '\\N' else float(row[1])
                        nVotes = None if row[2] == '\\N' else int(row[2]) 
                        if avgRating is not None and nVotes is not None:
                            cursor.execute("UPDATE movies SET avgRating_IMDB=?, numVotes_IMDB=? WHERE IMDB_id=?", (avgRating, nVotes, Imdb_id))
                            print(f'Updated numVotes and avgRating in ratingFile to {nVotes} and {avgRating} respectively with IMDB id {Imdb_id}')
                        else:
                            print('***!!!Could not update movie in ratingFile with IMDB id', Imdb_id)
        
        elif file_name == imdb_movie_file:
            with open(file_name, 'r', encoding='utf-8') as tsvfile:
                reader = csv.reader(tsvfile, delimiter='\t')
                next(reader)
                for row in tqdm(reader):
                    # Check if the 'Imdb_id' variable, with the first 2 letters removed and the rest converted to int, exists in the 'movies' table 
                    Imdb_id = int(row[0][2:])
                    cursor.execute("SELECT 1 FROM movies WHERE IMDB_id=?", (Imdb_id,))
                    data = cursor.fetchone()
                    if data is not None:
                        # If the movie exists in the 'movies' table, update the 'is_adult' and 'runtime_min' variables
                        is_adult = None if row[4] == '\\N' else int(row[4])
                        runtime_min = None if row[7] == '\\N' else int(row[7])
                        if is_adult is not None and runtime_min is not None:
                            cursor.execute("UPDATE movies SET is_adult=?, runtime_min=? WHERE IMDB_id=?", (is_adult, runtime_min, Imdb_id))
                            print(f'Updated is_adult and runtime_min in movieFile to {is_adult} and {runtime_min} respectively with IMDB id {Imdb_id}')
                        else:
                            print('***!!!Could not update movie in movieFile with IMDB id ', Imdb_id)


# Create and populate movie table using movies.csv, and links.csv, then add IMDB data 
def create_movie_table():
    with sqlite3.connect(database_name) as conn:
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS movies")
        cursor.execute('''CREATE TABLE movies
                            (movie_id INTEGER, IMDB_id INTEGER, title TEXT, genres TEXT, is_adult INTEGER, runtime_min INTEGER, 
                            avgRating_IMDB REAL, numVotes_IMDB INTEGER, avgRating_users REAL, numVotes_users INTEGER)''')
        with open(movie_file, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)
            for row in tqdm(reader):
                movie_id = int(row[0])
                title = row[1]
                genres = row[2]
                cursor.execute("INSERT INTO movies (movie_id, title, genres) VALUES (?, ?, ?)", (movie_id, title, genres))
        with open(link_file, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)
            for row in tqdm(reader):
                movie_id = int(row[0])
                Imdb_id = int(row[1])
                cursor.execute("UPDATE movies SET IMDB_id=? WHERE movie_id=?", (Imdb_id, movie_id))
        
    add_IMDB_data(imdb_rating_file)
    add_IMDB_data(imdb_movie_file)


# Create and populate ratings table using ratings.csv
def create_ratings_table():
    with sqlite3.connect(database_name) as conn:
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS ratings")
        cursor.execute('''CREATE TABLE ratings
                            (user_id INTEGER, movie_id INTEGER, rating REAL, timestamp INTEGER)''')
        with open(rating_file, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)
            for row in tqdm(reader):
                user_id = int(row[0])
                movie_id = int(row[1])
                rating = float(row[2])
                timestamp = int(row[3])
                cursor.execute("INSERT INTO ratings (user_id, movie_id, rating, timestamp) VALUES (?, ?, ?, ?)", (user_id, movie_id, rating, timestamp))


# Merge movies table and ratings table based on 'movie_id' variable, so that the ratings entries have all the movie's information, besides repeating the movie's id 
def merge_tables():
    with sqlite3.connect(database_name) as conn:
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS merged")
        cursor.execute('''CREATE TABLE merged AS
                          SELECT ratings.user_id, ratings.movie_id, ratings.rating, movies.title, 
                            movies.genres, movies.is_adult, movies.runtime_min, movies.release_date, 
                            movies.video_release_date, movies.avgRating_IMDB, movies.numVotes_IMDB, 
                            movies.avgRating_users, movies.numVotes_users, ratings.timestamp, movies.IMDB_id
                          FROM ratings
                          JOIN movies ON ratings.movie_id = movies.movie_id''')

# Print table name and number of entries in table, and the first 10 rows from the table 
def print_table_info(table_name):
    with sqlite3.connect(database_name) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM {}".format(table_name))
        print('The {} table has {} entries'.format(table_name, cursor.fetchone()[0]))
        cursor.execute("SELECT * FROM {} LIMIT 10".format(table_name))
        print(cursor.fetchall())                          

# Parrallel processing functions past this point (not done yet) 

# This function is run in a separate process.
# It reads and removes a row of data from the chunk if the movie does not exist in the 'movies' table
def process_data(chunk):
    results = []
    with sqlite3.connect(database_name) as conn:
        cursor = conn.cursor()
        reader = csv.reader(chunk, delimiter='\t')
        for row in reader:
            # Check if the 'Imdb_id' variable, with the first 2 letters removed and the rest converted to int, exists in the 'movies' table 
            Imdb_id = int(row[0][2:])
            cursor.execute("SELECT 1 FROM movies WHERE IMDB_id=?", (Imdb_id,))
            data = cursor.fetchone()
            # If if data is not None, add the movie to the results list
            if data is not None:
                results.append(row)            
    return results

def split_data_into_chunks(file_name, chunk_size=1000):
    # Read the entire file
    with open(file_name, 'r', encoding='utf-8') as f:
        data = f.readlines()

    # Split the data into chunks
    chunks = [data[i:i + chunk_size] for i in range(1, len(data), chunk_size)]

    return chunks

def update_moviesT_parrallel(file_name):
    # Split the data into chunks
    chunks = split_data_into_chunks(file_name)
    time_1 = time.time()
    print(f'The function took {time_1 - start_time} seconds to split the data into chunks.')

    # Create a pool of worker processes
    with Pool() as p:
        # Use the pool to process the data in parallel
        all_results = p.map(process_data, chunks)
    time_2 = time.time()
    print(f'The function took {time_2 - time_1} seconds to process the data in parallel.')

    # Write the results to the database
    with sqlite3.connect(database_name) as conn:
        cursor = conn.cursor()
        for results in all_results:
            for result in results:
                if file_name == imdb_movie_file:
                    # update movies table with result from process_data
                    Imdb_id = int(result[0][2:])
                    is_adult = None if result[4] == '\\N' else int(result[4])
                    runtime_min = None if result[7] == '\\N' else int(result[7])
                    if is_adult is not None or runtime_min is not None:
                        cursor.execute("UPDATE movies SET is_adult=?, runtime_min=? WHERE IMDB_id=?", (is_adult, runtime_min, Imdb_id))
                        print(f'Updated is_adult and runtime_min in movieFile to {is_adult} and {runtime_min} respectively with IMDB id {Imdb_id}')
                    else:
                        print('***!!!Could not update movie in movieFile with IMDB id ', Imdb_id)
                elif file_name == imdb_rating_file:
                    # update movies table with result from process_data
                    Imdb_id = int(result[0][2:])
                    avgRating = None if result[1] == '\\N' else float(result[1])
                    nVotes = None if result[2] == '\\N' else int(result[2])
                    if avgRating is not None or nVotes is not None:
                        cursor.execute("UPDATE movies SET avgRating_IMDB=?, numVotes_IMDB=? WHERE IMDB_id=?", (avgRating, nVotes, Imdb_id))
                        print(f'Updated numVotes and avgRating in ratingFile to {nVotes} and {avgRating} respectively with IMDB id {Imdb_id}')
                    else:
                        print('***!!!Could not update movie in ratingFile with IMDB id ', Imdb_id)
    
    time_3 = time.time()
    print(f'The function took {time_3 - time_2} seconds to update the database.')
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The file took {elapsed_time} seconds to upload.")


if __name__ == "__main__":

    update_moviesT_parrallel(imdb_movie_file)
    # print('Creating and populating movies table...')
    # with sqlite3.connect(database_name) as conn:
    #     cursor = conn.cursor()
    #     cursor.execute("SELECT * FROM movies WHERE IMDB_id=7193448")
    #     print(cursor.fetchall())    
    #create_ratings_table()
    #print_table_info('ratings')
    #create_movie_table()
    #print_table_info('movies')


    # merge_tables()
    # print_table_info('merged')