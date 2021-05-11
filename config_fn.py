import tweepy
import logging
import os
from os import environ

logger = logging.getLogger()

def create_twitter_api():
    consumer_key = "bHCoJVVE0KWTqmupsGiZ9Nqbq"
    consumer_secret = "QaFFjCkagludjJsBeGa29RGKBhhEHEU4YynZ0YNUnxwDQWBc1u"
    access_token = "3263024803-RCs752xN0UgOfTBHORjURpbqeVoKbetqrjyPrUD"
    access_token_secret = "yFFncnHntdOEtgjX9dmJY4fwuY2hQNoAGXFS6H5PkdrWU"


    #consumer_key = environ['twitter_consumer_key']
    #consumer_secret = environ['consumer_secret']
    #access_token =  environ['access_token']
    #access_token_secret = environ['access_token_secret']

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True,
        wait_on_rate_limit_notify=True)
    try:
        api.verify_credentials()
    except Exception as e:
        logger.error("Error creating API", exc_info=True)
        raise e
    logger.info("API created")
    return api

def s3():
    s3 = boto3.resource(
        service_name='s3',
        region_name='ap-south-1',
        aws_access_key_id=environ['aws_access_key_id'],
        aws_secret_access_key=environ['aws_secret_access_key'])
    return s3


def database_details():
    DATABASE_URL = 'postgresql+psycopg2://yqdhxhrzahgtxg:888b0d4cd6b957bb986a33ab8df3d3d9c81d8d2c6ff1462def9959384564af3f@ec2-54-216-185-51.eu-west-1.compute.amazonaws.com:5432/dbvk6d6g6njtfb'
    return DATABASE_URL

def database_details_web():
    DATABASE_URL = 'postgresql+psycopg2://yqdhxhrzahgtxg:888b0d4cd6b957bb986a33ab8df3d3d9c81d8d2c6ff1462def9959384564af3f@ec2-54-216-185-51.eu-west-1.compute.amazonaws.com:5432/dbvk6d6g6njtfb'
    return DATABASE_URL
