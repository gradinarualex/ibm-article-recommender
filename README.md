# **IBM Watson Article Recommender**
This project is builds a recommender system based on [IBM Watson](https://www.ibm.com/watson) users and articles provided by IBM through Udacity.
The recommender built provides two types of recommendations:
1. Knowledge based recommendations: for new users where we cannot assess preferences
2. User-based collaborative filtering recommendations: for existing users

## **Documentation**
Data used is user-article interaction, with no measure of appreciation (like/dislike).  
All recommendations are thus done on interactions regardless of reviews (as they are not available).  
Knowledge based recommendation uses top viewed articles by count of unique views.  
User-based collaborative filtering uses user similarity (vector dot product) and number of views per user to rank neighbors.

## **Installation**
Clone (or fork) this repository to your local machine, and create a virtual environment to install dependencises in `requirements.txt`:
```cli
$ pip install -r requirements.txt
```

If the next step doesn't work, make sure to generate clean data before using the recommender:  
```cli
$ python clean_data.py ./data/user-item-interactions.csv ./data/articles_community.csv

```

This will generate the clean datasets used to provide recommendations. They should be part of the repo, so this is a 'just in case'.

## **Usage**

After requirements are installed, you can use the recommender script to get recommendations for a user:
```cli
$ python recommender.py 5148 10
```
where `5148` is the user_id and `10` is the number of recommendations.
You should see results like this:
```cli
> Top 10 recommendations for user 5148:
> 1. use deep learning for image classification
> 2. insights from new york car accident reports
> 3. visualize car data with brunel
...
```


## **Licensing**
This project is licensed under MIT License - see the [LICENSE.md](LICENSE.md) file for details.
