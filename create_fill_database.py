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


"""
------------------------------------------------------------------------------------------------------------------------
    Functions to create and populate the database with csv files
------------------------------------------------------------------------------------------------------------------------
"""

# Create and populate movie table using movies.csv, and links.csv 
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
        
        # Create index on IMDB_id variable in movies table
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_movies_imdb_id ON movies (IMDB_id)")
        print('Created index on IMDB_id in movies table\n')


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
                            movies.genres, movies.is_adult, movies.runtime_min, movies.avgRating_IMDB, movies.numVotes_IMDB, 
                            movies.avgRating_users, movies.numVotes_users, ratings.timestamp, movies.IMDB_id
                          FROM ratings
                          JOIN movies ON ratings.movie_id = movies.movie_id''')
# Print table name and number of entries in table, and the first 10 rows from the table 
def print_table_info(table_name, num_rows=10):
    with sqlite3.connect(database_name) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM {}".format(table_name))
        print('The {} table has {} entries'.format(table_name, cursor.fetchone()[0]))
        cursor.execute("SELECT * FROM {} LIMIT {}".format(table_name, num_rows))
        print(cursor.fetchall())              
        print('\n')            
    return

def create_small_file(big_file, small_file, num_lines):
    with open(big_file, 'r', encoding='utf-8') as big, open(small_file, 'w', encoding='utf-8') as small:
        next(big)  # Skip the first line
        for _ in range(num_lines):
            line = big.readline()
            small.write(line)

"""
------------------------------------------------------------------------------------------------------------------------
 Parrallel processing functions to merge IMDB data with the movies table
------------------------------------------------------------------------------------------------------------------------
"""
# Reads and removes a row of data from the chunk if the movie does not exist in the 'movies' table
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

# Generator function to read data in chunks
def read_data_in_chunks(file_name, chunk_size=1000):
    chunk = []
    with open(file_name, 'r', encoding='utf-8') as f:
        next(f)  # Skip the first line
        for i, line in enumerate(f):
            if (i % chunk_size == 0 and i > 0):
                yield chunk
                chunk = []
            chunk.append(line)
        yield chunk  # yield the last chunk, which may be smaller than chunk_size

# Adds IMDB data to the movies table in parallel processes
def update_moviesT_parrallel(file_name, chunk_size=1000):
    
    start_time = time.time()
    
    # Create a pool of processes 
    with Pool() as p:
        # Use the pool to process the data in parallel
        line_count = 0
        for chunk in read_data_in_chunks(file_name, chunk_size):
            line_count += len(chunk)
            print(f'Processing lines {line_count - len(chunk) + 1} to {line_count}')
            results = p.map(process_data, [chunk])

            # Write the results to the database
            with sqlite3.connect(database_name) as conn:
                cursor = conn.cursor()
                for result in results[0]:
                    if file_name == imdb_movie_file:
                        # update movies table with result from process_data
                        Imdb_id = int(result[0][2:])
                        is_adult = None if result[4] == '\\N' else int(result[4])
                        runtime_min = None if result[7] == '\\N' else int(result[7])
                        if is_adult is not None or runtime_min is not None:
                            cursor.execute("UPDATE movies SET is_adult=?, runtime_min=? WHERE IMDB_id=?", (is_adult, runtime_min, Imdb_id))
                    
                    elif file_name == imdb_rating_file:
                        # update movies table with result from process_data
                        Imdb_id = int(result[0][2:])
                        avgRating = None if result[1] == '\\N' else float(result[1])
                        nVotes = None if result[2] == '\\N' else int(result[2])
                        if avgRating is not None or nVotes is not None:
                            cursor.execute("UPDATE movies SET avgRating_IMDB=?, numVotes_IMDB=? WHERE IMDB_id=?", (avgRating, nVotes, Imdb_id))
                conn.commit()
                print(f'Updated {len(results[0])} rows')

    print(f'Finished in {time.time() - start_time} seconds')


"""
------------------------------------------------------------------------------------------------------------------------
 Main function past this point to create and populate the database
------------------------------------------------------------------------------------------------------------------------
"""

def main():

    # write merged table to csv file
    with sqlite3.connect(database_name) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM merged")
        with open('merged.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow([i[0] for i in cursor.description])
            writer.writerows(cursor)
            
                    


    # create_small_file(imdb_movie_file, 'small_imdb_movie_file.tsv', 100000)
    # create_small_file(imdb_rating_file, 'small_imdb_rating_file.tsv', 100000)

    # update_moviesT_parrallel('small_imdb_movie_file.tsv')
    # update_moviesT_parrallel('small_imdb_rating_file.tsv')
    


    #print('Creating and populating ratings table...')
    #create_ratings_table()
    #print_table_info('ratings')

    #print('Creating and populating movies table...')
    #create_movie_table()
    #print_table_info('movies')

    #print('Adding IMDB data to movies table...')
    #update_moviesT_parrallel(imdb_rating_file)
    #update_moviesT_parrallel(imdb_movie_file)
    #print_table_info('movies')

    # print('Merging ratings and movies tables...')
    # merge_tables()
    # print_table_info('merged')

    print('Done!')


   


if __name__ == "__main__":

    main()
    
