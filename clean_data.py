# import libraries
import sys
import os
import re
import joblib
import pickle

import numpy as np
import pandas as pd


# define a user-to-mail mapper for annonymization
def email_mapper(df, col='email'):
    ''' This function creates a unique id for each email in a dataset
        to annonymize the users when performing recommendations. The
        mapping between email and user_id will be stored in a pickle
        file if needed.
    
        Args:
            df - (pandas dataframe) dataframe to map emails to unique id
            col - (string) column to annonymize data on
            
        Returns:
            email_encoded - (list) list of annonymized ids to user instead
                of emails
    
    '''
    
    coded_dict = dict() # initialize empty dictionary
    cter = 1 # initialize counter to 1
    email_encoded = [] # initialize list of encoded emails

    # for each email
    for val in df['email']:
        # if email not already encoded
        if val not in coded_dict:
            # encode email with current counter value
            coded_dict[val] = cter
            # increment counter value by 1
            cter += 1

        # add encoded value to list
        email_encoded.append(coded_dict[val])
        
    # save email mapping as pickle file
    with open('./email_encoding/encodings.pkl', 'wb') as file:
        pickle.dump(coded_dict, file)

    # return list of encoded values
    return email_encoded


def clean_data(user_item_path, articles_path):
    ''' Function used to load the data from a file using pandas,
        to extract the features of interest (V1-V7) and target and
        returns features as X and target as y.
        
        Args:
            user_item_path - (string) path to user-item data file
            articles_path - (string) path to article data file
            
        Returns:
            user_item_df - (pandas dataframe) clean dataframe of user-item interactions
            articles_df - (pandas dataframe) clean dataframe of article data
    '''
    
    # Load user-item data from file
    user_item_df = pd.read_csv(user_item_path)
    
    # Remove extra column
    del user_item_df['Unnamed: 0']
    # Convert `article_id` to int
    user_item_df['article_id'] = user_item_df['article_id'].astype(int)
    # Remove rows from user-item interaction with null user
    user_item_df = user_item_df[~user_item_df['email'].isnull()]
    # Get list of encodings and add column of user ids to the dataframe
    user_item_df['user_id'] = email_mapper(user_item_df)
    # Remove initial email column
    del user_item_df['email']
    
    # Load article data from file
    articles_df = pd.read_csv(articles_path)
    
    # Remove extra column
    del articles_df['Unnamed: 0']
    # Remove any rows that have the same article_id - only keep the first
    articles_df.drop_duplicates(subset=['article_id'], inplace=True)
    
    return user_item_df, articles_df


# create the user-article matrix with 1's and 0's
def create_user_item_matrix(df):
    ''' Return a matrix with user ids as rows and article ids on the columns
        with 1 values where a user interacted with an article and a 0 otherwise
    
        Args: 
            df - (pandas dataframe) with article_id, title, user_id columns
            
        Returns:
            user_item - user_item matrix
    '''
    
    # create a copy of the matrix (to not alter the original input)
    df_copy = df.copy(deep=True)
    
    # create view_count column to be able to pivot
    df_copy['view_count'] = 1
    # create pivot of user-item interactions with values as view count
    pivot = pd.pivot_table(df_copy, values='view_count', index=['user_id'], columns=['article_id'], aggfunc=np.sum)
    # convert values to 1 (if user has at least 1 interaction with the article) or 0 otherwise
    user_item = (pivot >= 1) * 1
    
    # return the user_item matrix
    return user_item 


def save_data(df, dataset_type):
    ''' Takes in a dataframe and a set name ('user-item' or 'item')
        and saves it as dataset_type + '_clean' in the data folder.
        
        Args:
            df (pandas dataframe) - dataframe to be saved as csv
            dataset_type (string) - 'user-item', 'item' or 'user-item-matrix'
                to define what dataset type it is
        
        Returns:
            None
    '''
    
    output_folder = './data/processed/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    if dataset_type == 'user-item':
        df.to_csv(output_folder + 'user-item-clean.csv', index=False, encoding='utf-8')

    elif dataset_type == 'item':
        df.to_csv(output_folder + 'item-clean.csv', index=False, encoding='utf-8')
        
    elif dataset_type == 'user-item-matrix':
        df.to_csv(output_folder + 'user-item-matrix.csv', index=False, encoding='utf-8')
        
    else:
        print('Error in save_data() function: dataset_type incorrect.\nFunction expects' +\
              'dataset_type to be either \'user-item\', \'item\' or \'user-item-matrix\'.')
              
    return None


def main():
    if len(sys.argv) == 3:
        user_item_path, articles_path = sys.argv[1:]
              
        print('1/3. Loading and cleaning data...\n   DATA FILE: {}\n    DATA FILE: {}'.format(user_item_path, articles_path), end='')
        user_item_df, articles_df = clean_data(user_item_path, articles_path)
        print('...done.')
              
        print('2/3. Creating user-item interaction matrix...', end='')
        user_item_matrix = create_user_item_matrix(user_item_df)
        print('...done.')
        
        data_dict = {
            'user-item': user_item_df,
            'item': articles_df,
            'user-item-matrix': user_item_matrix
        }
        
        print('3/3. Saving data to disk:')
        for dataset_type, dataset in data_dict.items():
            print('   DATA FILE: {}...'.format(dataset_type), end='')
            save_data(dataset, dataset_type)
            print('saved.')
        
    else:
        print('Please provide the filepath of the ibm user-item interactions dataset '\
              'as the first argument and the filepath of the articles dataset to '\
              'clean data and save it on disk. \n\nExample: python '\
              'clean_data.py ./data/user-item-interactions.csv ./data/articles_community.csv')
        

if __name__ == '__main__':
    main()