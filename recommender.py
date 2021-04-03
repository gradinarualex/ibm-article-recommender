# import libraries
import sys
import os
import joblib
import pickle

import numpy as np
import pandas as pd


def load_data():
    ''' Load clean data from the ./data/processed/ folder
        and return in the following order:
        1 - user_item_df
        2 - item_df
        3 - user_item_matrix
        
        Args:
            None
            
        Returns:
            user_item_df - (pandas dataframe) user-item interactions dataframe
            item_df - (pandas dataframe) item information dataframe
            user_item_matrix - (pandas dataframe) user-item interactions matrix
    '''
    
    user_item_df = pd.read_csv('./data/processed/user-item-clean.csv', encoding='utf-8')
    item_df = pd.read_csv('./data/processed/item-clean.csv', encoding='utf-8')
    user_item_matrix = pd.read_csv('./data/processed/user-item-matrix.csv', encoding='utf-8')
    
    return user_item_df, item_df, user_item_matrix


def get_article_names(article_ids, df):
    ''' Gets article name based on article id using df (interactions) dataframe
    
        Args:
            article_ids - (list) article ids
            df - (pandas dataframe) user-item interactions clean df
            
        Returns:
            article_names - (list) article names associated with the list
                of article ids (this is identified by the 'title' column)
    '''
    
    article_title_series = df[df['article_id'].isin(article_ids)]['title']
    article_names = article_title_series.unique()
    
    # return the article names associated with list of article ids
    return article_names


def get_top_articles(df, n):
    ''' Gets the user-to-article interaction and
        return list of top n articles viewed by users
    
        Args:
            df - (pandas dataframe) user-item interactions clean df
            n - (int) the number of top articles to return
            
        Returns:
            top_articles - (list) the top 'n' article titles
    
    '''
    # count number of views per article id
    views_per_article = df.groupby(['article_id'], as_index=False)\
                            ['user_id'].count()\
                            .sort_values(by='user_id', ascending=False)
    
    # get top n articles by views
    top_n_views_per_article = views_per_article.iloc[:n, :]
    # get list of ids of top n articles
    top_article_ids = top_n_views_per_article['article_id'].tolist()
    
    # get list of unique articles and titles
    articles_df = df[['article_id', 'title']].drop_duplicates().set_index('article_id')['title']
    # get top article titles
    top_articles = articles_df[top_article_ids].tolist()
    
    # return the top articles titles from df (not df_content)
    return top_articles


def is_new_user(user_id, df):
    ''' Check if user exists in the user-item interactions dataframe
    
        Args:
            user_id - (int) user_id to serach in df
            df - (pandas dataframe) user-item interactions clean df
            
        Returns:
            is_new - (bool) True value if user is new, else False
    '''
    
    # check if user_id exists in 'user_id' column of the user-item interactions
    is_new = np.isin(user_id, df['user_id'])
    
    # return the bool value of our result
    return bool(is_new)


def get_user_articles(user_id, user_item):
    ''' Provides a list of the article_ids and article titles that
        have been seen by a user
        
        Args:
            user_id - (int) a user id
            user_item - (pandas dataframe) matrix of users by articles:
                1's when a user has interacted with an article, 0 otherwise
                
        Returns:
            article_ids - (list) article ids seen by the user
            article_names - (list) article names associated with the list of
                article ids (this is identified by the doc_full_name column
                in df_content)    
    '''
    
    # get user row
    user_views = user_item.loc[user_id, :]
    # get list of article_ids the user viewed
    article_ids = user_views[user_views == 1].index.tolist()
    # get name of each article_id
    article_names = get_article_names(article_ids)
    
    return article_ids, article_names


def get_top_sorted_users(user_id, df, user_item):
    ''' Uses user-item interaction matrix and interactions df
        to generate a neighbor_df dataframe for a particular user.
        This is used to measure similarity on more than just dot
        product, but also on number of interactions.
    
        Args:
            user_id - (int) user id
            df - (pandas dataframe) user-item interactions clean df
            user_item - (pandas dataframe) matrix of users by articles: 1's
                when a user has interacted with an article, 0 otherwise
                
        Returns:
            neighbors_df - (pandas dataframe) a dataframe with:
                neighbor_id - is a neighbor user_id
                similarity - measure of the similarity of each user to the provided user_id
                num_interactions - the number of articles viewed by the user
                
        Notes:
        * sort the neighborhood_df by similarity and then by number of interactions where
        highest of each is higher in the dataframe
    
    '''
    
    # compute similarity of each user to the provided user
    user_similarity = user_item.dot(user_item.loc[1, :])
    
    # convert result into dataframe
    user_similarity = user_similarity.reset_index()
    
    # change value column name to 'similarity' 
    user_similarity = user_similarity.rename({0: 'similarity'}, axis=1)
    
    # get number of interactions by user
    user_interactions = df.groupby(['user_id'])['article_id']\
                            .count()\
                            .reset_index()\
                            .rename({'article_id': 'num_interactions'}, axis=1)
    
    # join the two datasets to get both similarity and num_interactions in
    # the same dataframe
    neighbors_df = user_similarity.merge(user_interactions, how='left', on='user_id')
    
    # rename user_id column to 'neighbor_id'
    neighbors_df = neighbors_df.rename({'user_id': 'neighbor_id'}, axis=1)
    
    # sort dataframe
    neighbors_df = neighbors_df.sort_values(['similarity', 'num_interactions'], ascending=[False, False])
    
    # remove rows where neighbor_id == user_id
    neighbors_df = neighbors_df[neighbors_df['neighbor_id'] != user_id]
    
    # return the dataframe specified in the doc_string
    return neighbors_df


def user_user_recs(user_id, df, m=10):
    ''' User based collaborative filtering. Loops through the users based
        on closeness to the input user_id. For each user - finds articles
        the user hasn't seen before and provides them as recs. Does this
        until m recommendations are found.

        Args:
            user_id - (int) a user id
            df - (pandas dataframe) user-item interactions clean df
            m - (int) the number of recommendations you want for the user
    
        Returns:
            recs - (list) a list of recommendations for the user by article id
            rec_names - (list) a list of recommendations for the user by article title
    
        Notes:
        * choose the users that have the most total article interactions 
        before choosing those with fewer article interactions.
        * choose articles with the most total interactions before choosing
        those with fewer total interactions. 
   
    '''
    
    # create a list of articles read by the user
    seen_article_ids, _ = get_user_articles(user_id)
    
    # count views per article for each article and store in a sorted series
    article_views = df.groupby(['article_id'], as_index=False)['user_id'].count()
    # rename values column to 'view_count'
    article_views = article_views.rename({'user_id': 'view_count'}, axis=1)
    # sort values from highest to lowest view_count
    article_views = article_views.sort_values(by='view_count', ascending=False)
    
    # generate neighbors dataframe
    neighbors_df = get_top_sorted_users(user_id)
    
    # create empty list of recos and reco names to be filled in
    recs = []
    rec_names = []
    
    # for each similar user in list
    for neighbor_id in neighbors_df['neighbor_id'].tolist():
        # get list of article_ids
        article_ids, _ = get_user_articles(neighbor_id)
        
        # filter out articles the user has already seen
        user_reco_ids = [item for item in article_ids if item not in seen_article_ids]
        
        # get articles recommended to user
        recommended_articles = article_views[article_views['article_id'].isin(user_reco_ids)]
        # get list of article ids
        user_recos = recommended_articles['article_id'].tolist()
        # get list of article names
        user_reco_names = get_article_names(user_recos)
        
        # add user_recos and reco_names to list
        recs.extend(user_recos)
        rec_names.extend(user_reco_names)
        
        # if number of recommendations exceeds limit, stop
        if len(recs) >= m:
            break
    
    return recs[:m], rec_names[:m]


def main():
    if len(sys.argv) == 3:
        user_id, reco_count = sys.argv[1:]
        
        # convert arguments to int
        user_id = int(user_id)
        reco_count = int(reco_count)
        
        # read in clean data from ./data/processed/ folder
        # if no data in processed folder, run first the clean_data.py
        # script
        user_item_df, item_df, user_item_matrix = load_data()
        
        if is_new_user(user_id, user_item_df):
            recommendations = get_top_articles(user_item_df, reco_count)
        
        else:
            recommendations = user_user_recs(user_id, user_item_df, reco_count)

        print('Top {} recommendations for user {}:'.format(reco_count, user_id))
        for idx, reco in enumerate(recommendations):
            print('{}. {}'.format(idx + 1, reco))
        
        return recommendations
        
    else:
        print('Please provide a single user_id to make recommendations to'\
              '\nExample: python '\
              'recommender.py 5148 10')
        

if __name__ == '__main__':
    main()