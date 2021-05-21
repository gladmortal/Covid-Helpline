import tweepy
import logging
from config_fn import *
import time
from data_utils import *
import pandas as pd
import os.path
import os
import math
import re
from collections import Counter
import boto3
from os import path
from os import environ
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime




logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()



def twitter_data():
    api = create_twitter_api()
    since_id = 1
    MAX_TWEETS = 150
    while True:
        print('data extraction from twitter.....')

        ##### Functions Execution to extract from source (Twitter), Extract Keywords, Resources Filter, City, Requirement Extraction ######
        df_raw_tweet,since_id = tweet_extraction(api,since_id)

        count_row_raw = df_raw_tweet.shape[0]
        if count_row_raw > 0:
            df_tweets_with_keywords = get_keywords(df_raw_tweet,'tweet','cleaned_tweet_tokens')
            list_of_keywords_providing_help = list_to_lower(list_of_keywords_providing_help_raw)
            list_of_keywords_asking_help = list_to_lower(list_of_keywords_asking_help_raw)
            df_tweets_only_resource = keep_resource_only(df_tweets_with_keywords,'tweet','is_resource')
            df_tweets_only_resource['tweet'] = df_tweets_only_resource['tweet'].apply(cleaned_tweet)

            count_row_df_tweets_only_resource = df_tweets_only_resource.shape[0]
            if count_row_df_tweets_only_resource > 0:

                df_tweets_with_city = city_extraction(df_tweets_only_resource, 'cleaned_tweet_tokens','city')
                df_tweets_with_requirement = requirement_extraction(df_tweets_with_city, 'cleaned_tweet_tokens',list_of_keywords_providing_help,'requirement')
                df_tweets_with_requirement = df_tweets_with_requirement[df_tweets_with_requirement['requirement'].notna()]



                ## Code for Tech Validation

                ###-------Need to Work on ------------###

                #### Resetting and Dropping the index column before inserting in database #####
                df_tweets_with_requirement.reset_index(drop=True, inplace=True)
                df_tweets_with_requirement.drop(['index'], axis = 1,inplace=True)

                ## Adding Additional Columns ###########
                df_tweets_with_requirement["current_time"] = datetime.utcnow().strftime("%Y-%m-%d, %H:%M:%S")
                df_tweets_with_requirement["extra_details"] = ""
                df_tweets_with_requirement["auto_validation_score"] = ""
                df_tweets_with_requirement["validation_status"] = ""
                df_tweets_with_requirement["validated_by"] = ""
                df_tweets_with_requirement["validation_details"] = ""

                df_tweets_with_requirement["source"] = "Twitter"
                df_tweets_with_requirement['phone_number']=df_tweets_with_requirement['tweet'].str.extract(r'(\+?\d[\d -]{8,12}\d)')  ### Regex can take more improvement


                #### Removing, Renaming column names before inserting in database. #######

                cols = ['tweet_id', 'tweet', 'user','creation_time','followers_count','current_time','source','cleaned_tweet_tokens', 'city', 'requirement',
                'phone_number','extra_details','auto_validation_score','validation_status','validated_by','validation_details']
                df_tweets_with_requirement = df_tweets_with_requirement[cols]
                df_tweets_with_requirement.rename(columns = {'tweet_id' : 'id','tweet' : 'message', 'user' : 'provider','creation_time': 'time','followers_count' : 'followers_count',
                'current_time': 'time_inserted','source':'source','cleaned_tweet_tokens':'cleaned_tweet_tokens', 'city':'city', 'requirement':'requirement_list','phone_number':'phone_number',
                'extra_details':'extra_details','auto_validation_score':'auto_validation_score','validation_status':'validation_status','validated_by':'validated_by','validation_details':'validation_details'}, inplace = True)

                #####  Inserting into the database ###########
                engine = create_engine(database_details())
                df_tweets_with_requirement.to_sql('covid_resource_details', engine,if_exists='append', method=psql_insert_copy,index=False)
                time.sleep(120)
            else:
                time.sleep(120)
        else:

        #exit()
        #if path.exists(final_data_to_reply_file):
        #    s3.Bucket('twitter-covid-help-data').upload_file(Filename=final_data_to_reply_file, Key=final_data_to_reply_file)
        #else:
        #    pass
            time.sleep(120)


twitter_data()
