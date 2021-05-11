#!/usr/bin/env python
# coding: utf-8

# In[97]:


import tweepy
import csv
import pandas as pd
import numpy as np # linear algebra
import spacy
nlp = spacy.load('en_core_web_sm')
#nlp_large = spacy.load('en_core_web_lg')
from collections import Counter
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stoplist = set(stopwords.words("english"))
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import string
from spacy.lang.en import English
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from gensim.models import doc2vec
from sklearn import utils
import gensim
from gensim.models.doc2vec import TaggedDocument
import re
from textblob import TextBlob
from spellchecker import SpellChecker
import pickle
import re
from collections import Counter
import math
import boto3
import tweepy
import logging
import time
import os.path
import math
from os import path
from datetime import datetime, timedelta
from collections import Counter
from datetime import timezone
import streamlit
from os import environ
from sqlalchemy import create_engine
import psycopg2
import io
from io import StringIO
from config_fn import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()



spell = SpellChecker()


## Adding some extra stopwords
model_filename = 'finalized_model.sav'
parser = English()
stoplist.add('...')
stoplist.add("'s")
stoplist.add('amp')
stoplist.add('rt')

#### static Keyword list

list_of_keywords_providing_help_raw = ['Oxygen','Oxigen','bed','oxygen','hospital','plasma','icu','injection','refill','remdesivir'
                                   ,'bedsneeded','lung','pneumonia','tocilizumab','chest','ambulance'
                                   ,'ventilator','o2','remdisivir','oxygencylinder','fabiflu','covidbeds'
                                   ,'drug','ventilatorbed','blood','plasmadonor','remidesivir','medazolan'
                                   ,'remdivisir','concentrator','cylinder','medicine','400mg','fabi','temi','temiflu','Dexamethasone','OxygenCylinders'
                                   'Actemra','Remidivisir','oxygenrefill','Baricitinib','favipiravir','favilavir','ambulence','ambulanse','ambulense','HospitalBeds','amulance']

list_of_keywords_asking_help_raw = ['friend','admit','serious','Emergency','relative'
                                ,'Father','Mother','Brother','Sister','Uncle','Grandfather','GrandMother'
                                ,'Husband','Wife','Family','look','urgent','Aunt','Aunty','please','spo2','critical','severe','appreciate','level']



def tweet_extraction(api,since_id):

    raw_data_list = []
    new_since_id = since_id
    print(new_since_id)
    for tweet_info in tweepy.Cursor(api.search, q='#CovidHelp OR #CovidResources OR #VerifiedCovidResources',lang='en', count=100,result_type='latest',tweet_mode='extended',since_id=new_since_id).items(150):
        #print(new_since_id)
        isRT = hasattr(tweet_info, 'retweeted_status')
        timestr = time.strftime("%Y%m%d-%H%M%S")
        file_name = 'tweets_extracted_'+timestr+'.csv'
        tweet_i=tweet_info.full_text
        own_tweet = 'Response from Bot'
        if isRT == False and own_tweet not in tweet_i:
            new_since_id = max(tweet_info.id, new_since_id)
            #if not tweet_info.retweeted_status:
            creation_time = tweet_info.created_at
            if 'retweeted_status' in dir(tweet_info):
                tweet_id = tweet_info.id
                tweet=tweet_info.retweeted_status.full_text
                user=tweet_info.retweeted_status.user.screen_name
                creation_time = tweet_info.retweeted_status.created_at
            else:
                tweet_id = tweet_info.id
                tweet=tweet_info.full_text
                user=tweet_info.user.screen_name
                creation_time = tweet_info.created_at
            user_details = api.get_user(user)
            followers_count = user_details.followers_count
            tweet_details = [tweet_id,creation_time,tweet,user,followers_count]
            raw_data_list.append(tweet_details)
    df_raw = pd.DataFrame(raw_data_list,columns=['tweet_id','creation_time','tweet','user','followers_count'])
    return df_raw,new_since_id



def data_from_s3(s3):
    file_name = "last_few_hours_data.csv"
    if not path.exists(file_name):
        with open(file_name, "w") as cv:
            file_time = 1620050896
    else:
        file_time = os.path.getmtime(file_name)
    for obj in s3.Bucket('twitter-covid-help-data').objects.all():
        last_modified_s3_file = obj.last_modified
        timestamp = last_modified_s3_file.replace(tzinfo=timezone.utc).timestamp()
        if timestamp > file_time:
            obj = s3.Bucket('twitter-covid-help-data').Object(obj.key).get()
            #print(obj)
            foo = pd.read_csv(obj['Body'], index_col=None)
            foo.to_csv(file_name,mode='a',header=False, index=False)
    return file_name

def cleaned_tweet(x):
    tweet = x
    emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    tweet = emoji_pattern.sub(r'', tweet) # no emoji
    tweet = re.sub(r"@\w+"," ", tweet)  # Removing Mentions


    ###### Cleaning ###########
    tweet = re.sub(r"\n", " ", tweet)
    tweet = re.sub(r"#CovidHelp", "", tweet)
    tweet = re.sub(r"#DelhiFightsCorona", "", tweet)
    tweet = re.sub(r"#CovidSOSIndia", "", tweet)
    tweet = re.sub(r"#COVIDEmergencyIndia", "", tweet)
    tweet = re.sub(r"#CovidIndiaInfo", "", tweet)
    tweet = re.sub(r"#COVIDEmergencyIndia", "", tweet)
    print(tweet)
    return tweet


def extra_clean(tweet):

    tweet = re.sub(r"there's", "there is", tweet)
    tweet = re.sub(r"We're", "We are", tweet)
    tweet = re.sub(r"That's", "That is", tweet)
    tweet = re.sub(r"won't", "will not", tweet)
    tweet = re.sub(r"they're", "they are", tweet)
    tweet = re.sub(r"Can't", "Cannot", tweet)
    tweet = re.sub(r"wasn't", "was not", tweet)
    tweet = re.sub(r"aren't", "are not", tweet)
    tweet = re.sub(r"isn't", "is not", tweet)
    tweet = re.sub(r"What's", "What is", tweet)
    tweet = re.sub(r"haven't", "have not", tweet)
    tweet = re.sub(r"hasn't", "has not", tweet)
    tweet = re.sub(r"There's", "There is", tweet)
    tweet = re.sub(r"He's", "He is", tweet)
    tweet = re.sub(r"It's", "It is", tweet)
    tweet = re.sub(r"You're", "You are", tweet)
    tweet = re.sub(r"I'M", "I am", tweet)
    tweet = re.sub(r"shouldn't", "should not", tweet)
    tweet = re.sub(r"wouldn't", "would not", tweet)
    tweet = re.sub(r"i'm", "I am", tweet)
    tweet = re.sub(r"I'm", "I am", tweet)
    tweet = re.sub(r"Isn't", "is not", tweet)
    tweet = re.sub(r"Here's", "Here is", tweet)
    tweet = re.sub(r"you've", "you have", tweet)
    tweet = re.sub(r"we're", "we are", tweet)
    tweet = re.sub(r"what's", "what is", tweet)
    tweet = re.sub(r"couldn't", "could not", tweet)
    tweet = re.sub(r"we've", "we have", tweet)
    tweet = re.sub(r"would've", "would have", tweet)
    tweet = re.sub(r"it'll", "it will", tweet)
    tweet = re.sub(r"we'll", "we will", tweet)
    tweet = re.sub(r"We've", "We have", tweet)
    tweet = re.sub(r"he'll", "he will", tweet)
    tweet = re.sub(r"Y'all", "You all", tweet)
    tweet = re.sub(r"Weren't", "Were not", tweet)
    tweet = re.sub(r"Didn't", "Did not", tweet)
    tweet = re.sub(r"they'll", "they will", tweet)
    tweet = re.sub(r"they'd", "they would", tweet)
    tweet = re.sub(r"DON'T", "DO NOT", tweet)
    tweet = re.sub(r"they've", "they have", tweet)
    tweet = re.sub(r"i'd", "I would", tweet)
    tweet = re.sub(r"should've", "should have", tweet)
    tweet = re.sub(r"we'd", "we would", tweet)
    tweet = re.sub(r"i'll", "I will", tweet)
    tweet = re.sub(r"weren't", "were not", tweet)
    tweet = re.sub(r"They're", "They are", tweet)
    tweet = re.sub(r"let's", "let us", tweet)
    tweet = re.sub(r"it's", "it is", tweet)
    tweet = re.sub(r"can't", "cannot", tweet)
    tweet = re.sub(r"don't", "do not", tweet)
    tweet = re.sub(r"you're", "you are", tweet)
    tweet = re.sub(r"i've", "I have", tweet)
    tweet = re.sub(r"that's", "that is", tweet)
    tweet = re.sub(r"i'll", "I will", tweet)
    tweet = re.sub(r"doesn't", "does not", tweet)
    tweet = re.sub(r"i'd", "I would", tweet)
    tweet = re.sub(r"didn't", "did not", tweet)
    tweet = re.sub(r"ain't", "am not", tweet)
    tweet = re.sub(r"you'll", "you will", tweet)
    tweet = re.sub(r"I've", "I have", tweet)
    tweet = re.sub(r"Don't", "do not", tweet)
    tweet = re.sub(r"I'll", "I will", tweet)
    tweet = re.sub(r"I'd", "I would", tweet)
    tweet = re.sub(r"Let's", "Let us", tweet)
    tweet = re.sub(r"you'd", "You would", tweet)
    tweet = re.sub(r"It's", "It is", tweet)
    tweet = re.sub(r"Ain't", "am not", tweet)
    tweet = re.sub(r"Haven't", "Have not", tweet)
    tweet = re.sub(r"Could've", "Could have", tweet)
    tweet = re.sub(r"youve", "you have", tweet)


    # Hashtags and usernames


    # Urls
    tweet = re.sub(r"https?:\/\/t.co\/[A-Za-z0-9]+", "", tweet)

    # Words with punctuations and special characters
    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`"
    for p in punctuations:
        tweet = tweet.replace(p, f' {p} ')

    # ... and ..
    tweet = tweet.replace('...', ' ... ')
    if '...' not in tweet:
        tweet = tweet.replace('..', ' ... ')


    return tweet

# In[16]:


def cleanup_text(docs, logging=False):
    texts = []
    counter = 1
    for doc in docs:
        doc = extra_clean(doc)
        if counter % 1000 == 0 and logging:
            print("Processed %d out of %d documents." % (counter, len(docs)))
        counter += 1
        doc = nlp(doc, disable=['parser', 'ner'])
        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-' and not tok.is_punct]
        #tokens = [str(TextBlob(tok).correct()) for tok in tokens]
        #tokens = [spell.correction(tok) for tok in tokens]
        tokens = [tok for tok in tokens if tok not in stoplist and len(tok) > 1 and not tok.startswith("@") and not tok.startswith("http") ]
        tokens = ' '.join(tokens)
        texts.append(tokens)
    return pd.Series(texts)


# In[17]:


def get_keywords(dataframe,column_name,new_column_name):
    cleaned_tweet = (cleanup_text(dataframe[column_name]))
    print(cleaned_tweet)
    #print('This')
    dataframe[new_column_name] = cleaned_tweet.str.split()
    return dataframe

def get_keyword_frequency(dataframe,column_name):
    cleaned_tweet = (cleanup_text(dataframe[column_name]))
    words = cleaned_tweet.str.split()
    word_counts = pd.value_counts(words.apply(pd.Series).stack())
    #display(word_counts.to_string())

def list_to_lower(list_to_input):
    return list(map(lambda x:x.lower(),list_to_input))


# In[18]:

def keep_resource_only(dataframe,column_name,new_column_name):
    model_from_pickle = pickle.load(open(model_filename, 'rb'))
    predicted = model_from_pickle.predict(cleanup_text(dataframe[column_name]))
    dataframe[new_column_name] = predicted
    dataframe = dataframe[dataframe[new_column_name] != 3]
    dataframe = dataframe[dataframe[new_column_name] != 1]
    dataframe.reset_index(inplace = True)
    return dataframe

def help_or_resource(dataframe,column_name,list_for_lookup,new_column_name):
    status_list = []
    for i in dataframe[column_name]:
        status = (bool(set(i) & set(list_for_lookup)))
        status_list.append(status)
        #tweets_extracted['question'].loc[i] = status
    dataframe['question'] = status_list
    return dataframe


# In[19]:


def city_extraction(dataframe,column_name,new_column_name):
    crimefile = open('citylist.txt', 'r')
    reader = csv.reader(crimefile)
    allRows = [row for row in reader]
    flat_list = [item for sublist in allRows for item in sublist]
    city_list = list(map(lambda x:x.lower(),flat_list))
    city_check_list = []
    for i in dataframe[column_name]:
        city = list(set(i).intersection(city_list))
        city = (' '.join(city))
        city_check_list.append(city)
    dataframe[new_column_name] = city_check_list
    return dataframe


# In[20]:


def requirement_extraction(dataframe,column_name,list_for_lookup,new_column_name):
    requirement_list = []
    for i in dataframe[column_name]:
        requirement = list(set(i).intersection(list_for_lookup))
        requirement = (' '.join(requirement))
        requirement_list.append(requirement)
    dataframe[new_column_name] = requirement_list
    return dataframe


# In[24]:

def export_to_file(dataframe,file_name):
    dataframe.to_csv(file_name, index=False)


def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def text_to_vector(text):
    words = re.compile(r"\w+").findall(text)
    return Counter(words)

def matching(tweets_extracted):
    df1,df2 = (tweets_extracted.groupby('question'))
    #print(df1)
    if df1[0] == False:
        answer = df1[1]
    if df2[0] == True:
        question = df2[1]
    #print(question)
    #print(question['tweet_id'])
    #print(answer['tweet_id'])
    #question = question[question['tweet']]
    #question.reset_index(inplace = True)
    file_name = 'final_output_to_use.csv'
    with open(file_name, "w",newline='') as fp:
        wr = csv.writer(fp,lineterminator='\n')
        header = ['tweet_id_requester','time_requester','tweet_requester','username_requester',
        'keywords_requester','question_flag_requester','city_requester','requirement_requester',
        'tweet_id_provider','time_provider','tweet_provider','username_provider','keywords_provider',
        'question_flag_provider','city_provider','requirement_provider','score','city_match','response_id']
        wr.writerow(header)
        for index, question_row in question.iterrows():
            #print(index)
            #print(question)
            for index, answer_row in answer.iterrows():
                vector1 = text_to_vector(' '.join(question_row['requirement_list']))
                vector2 = text_to_vector(' '.join(answer_row['requirement_list']))
                score = get_cosine(vector1, vector2)
                #print(score)
                #print(type(question_row))
                #question_req = nlp(' '.join(question_row['requirement_list']))
                #answer_req = nlp(' '.join(answer_row['requirement_list']))
                #score = (question_req.similarity(answer_req))
                #score = cosine_similarity(question_row['requirement_list'], answer_row['requirement_list'])
                city_match = list(set(question_row['city']).intersection(answer_row['city']))
                #print(city_match)
                #print((question_row['requirement_list']))
                #print(vector1)
                #print(vector2)
                #print(answer_row['user'])
                if score > 0.6 and len(city_match) > 0 and len(question_row['requirement_list']) > 0:
                    q_list = question_row.tolist()
                    a_list = answer_row.tolist()
                    del q_list[0]  # Removing dataframe index
                    del a_list[0] # Removing dataframe index
                    data_to_reply = q_list + a_list
                    data_to_reply.append(score)
                    data_to_reply.append(city_match)
                    response_id = str(q_list[0]) + str(a_list[0])
                    data_to_reply.append(response_id)
                    print('If any match...')
                    print(data_to_reply)
                    wr.writerow(data_to_reply)
        fp.close()
        return file_name


def psql_insert_copy(table, conn, keys, data_iter):
    # gets a DBAPI connection that can provide a cursor
    dbapi_conn = conn.connection
    with dbapi_conn.cursor() as cur:
        s_buf = StringIO()
        writer = csv.writer(s_buf)
        writer.writerows(data_iter)
        s_buf.seek(0)

        columns = ', '.join('"{}"'.format(k) for k in keys)
        if table.schema:
            table_name = '{}.{}'.format(table.schema, table.name)
        else:
            table_name = table.name

        sql = 'COPY {} ({}) FROM STDIN WITH CSV'.format(
            table_name, columns)
        cur.copy_expert(sql=sql, file=s_buf)



def insert_data_in_table(data_to_insert):
    engine = create_engine('postgresql+psycopg2://yqdhxhrzahgtxg:888b0d4cd6b957bb986a33ab8df3d3d9c81d8d2c6ff1462def9959384564af3f@ec2-54-216-185-51.eu-west-1.compute.amazonaws.com:5432/dbvk6d6g6njtfb')
    data_to_insert.head(0).to_sql('covid_resource_details', engine, if_exists='append',index=False)
    conn = engine.raw_connection()
    cur = conn.cursor()
    output = io.StringIO()
    data_to_insert.to_csv(output, sep='\t', header=False, index=False)
    print(data_to_insert)
    output.seek(0)
    contents = output.getvalue()
    cur.copy_from(output, 'covid_resource_details', null="") # null values become ''
    conn.commit()
    return('True')

def new_matching(data_to_match_from,input_frame):
    answer = data_to_match_from
    #print(tweets_extracted_from_input)
    #df1,df2 = (tweets_extracted_from_input.groupby('question'))
    #print(df1)
    #if df1[0] == False:
#        answer = df1[1]
    #if df2[0] == True:
#        NA = df2[1]
    print('--------------------NEW MATCHING------------------------')
    print(answer.columns)
    data_to_reply_full = []
    for index, question_row in input_frame.iterrows():
        #print(index)
        #print(question_row)
        for index, answer_row in answer.iterrows():
            data_to_reply = []
            print(type(answer_row['requirement_list']))
            print(type(answer_row['city']))

            if(answer_row['requirement_list'] is not None and answer_row['city'] is not None):
                a_r = list(answer_row['requirement_list'].split(" "))
                a_c = list(answer_row['city'].split(" "))
                vector1 = text_to_vector(question_row['requirement_list'])
                vector2 = text_to_vector(answer_row['requirement_list'])
                score = get_cosine(vector1, vector2)
                city_match = list(set(question_row['city'].split(" ")).intersection(a_c))
                if score > 0.6 and len(city_match) > 0 and len(question_row['requirement_list']) > 0:
                    q_list = question_row.tolist()
                    a_list = answer_row.tolist()
                    del q_list[0]  # Removing dataframe index
                    #del a_list[0] # Removing dataframe index
                    data_to_reply = a_list
                    data_to_reply.append(score)
                    data_to_reply.append(city_match)
                    response_id = str(q_list[0]) + str(a_list[0])
                    data_to_reply.append(response_id)
                    print(data_to_reply)
                    print('########################')
                    data_to_reply_full.append(data_to_reply)
                    print('If any match...')
                    #data_to_reply_full = pd.DataFrame(data_to_reply_full, columns = ['tweet_id', 'time','tweet','user','cleaned_tweet_tokens','question_flag_provider','city_provider','keywords_provider','score','city_match','response_id'])
                    print(data_to_reply_full)
            else:
                continue
        data_to_reply_full_df = pd.DataFrame(data_to_reply_full, columns = ['id','message','provider','time','followers_count','time_inserted','source',
        'cleaned_tweet_tokens','city','requirement_list','phone_number','extra_details','auto_validation_score','validation_status','validated_by','validation_details',
        'validated','score','city_match','response_id'])
        return data_to_reply_full_df



def tweet_processing(file_name):
    tweets_extracted = pd.read_csv(file_name,names=["tweet_id", "date", "tweet", "user"])
    tweets_extracted.drop_duplicates(subset=['tweet_id'],inplace=True,ignore_index=True)
    #display(tweets_extracted)
    tweets_extracted = get_keywords(tweets_extracted,'tweet','cleaned_tweet_tokens')
    #get_keyword_frequency()    #### Optinal Function to run and see the frequency of keywords
    list_of_keywords_providing_help = list_to_lower(list_of_keywords_providing_help_raw)
    list_of_keywords_asking_help = list_to_lower(list_of_keywords_asking_help_raw)
    #print(tweets_extracted)
    #tweets_extracted = help_or_resource(tweets_extracted,'cleaned_tweet_tokens',list_of_keywords_asking_help,'question')
    tweets_extracted = help_or_resource_new(tweets_extracted,'tweet','question')
    #print(tweets_extracted)
    tweets_extracted = city_extraction(tweets_extracted, 'cleaned_tweet_tokens','city')
    tweets_extracted = requirement_extraction(tweets_extracted, 'cleaned_tweet_tokens',list_of_keywords_providing_help,'requirement')
    #print(tweets_extracted)
    #file_name = 'processed_data.csv'
    #tweets_extracted.to_csv(file_name,header=False, index=False)
    return tweets_extracted
    #export_to_file(tweets_extracted,'output_data.csv')

def input_processing(data_to_match_from,sentence):
    #tweets_extracted = pd.read_csv(file_name,names=["tweet_id", "date", "tweet", "user"])
    #display(tweets_extracted)
    data = {'tweet':[sentence]}
    df_input = pd.DataFrame(data)
    df_input = get_keywords(df_input,'tweet','cleaned_tweet_tokens')
    list_of_keywords_providing_help = list_to_lower(list_of_keywords_providing_help_raw)
    list_of_keywords_asking_help = list_to_lower(list_of_keywords_asking_help_raw)
    #df_input = help_or_resource_new(df_input,'tweet','question')
    #print(df_input)
    df_input = city_extraction(df_input, 'cleaned_tweet_tokens','city')
    df_input = requirement_extraction(df_input, 'cleaned_tweet_tokens',list_of_keywords_providing_help,'requirement_list')
    #print(df_input)
    #file_name = 'processed_data.csv'
    final_data_to_display = new_matching(data_to_match_from,df_input)
    return final_data_to_display
    #export_to_file(tweets_extracted,'output_data.csv')

# In[63]:


def t_main(sentense):

    alchemyEngine   = create_engine(database_details());
    dbConnection    = alchemyEngine.connect();
    # Read data from PostgreSQL database table and load into a DataFrame instance
    data_from_table      = pd.read_sql("select * from \"covid_resource_details\" where Time BETWEEN NOW() - INTERVAL '48 HOURS' AND NOW()", dbConnection);
    #print(data_to_match_from)
    print('Look........')
    data_to_return = input_processing(data_from_table,sentense)
    if isinstance(data_to_return, list):
        return data_to_return
    else:
        data_to_return = data_to_return.astype({"id": str})
        data_to_return['link'] = data_to_return['id'].apply(lambda x: f"https://twitter.com/covidhelp/status/{x}")
        data_to_return['time'] = pd.DatetimeIndex(data_to_return['time']) + timedelta(hours=5,minutes=30)
        data_to_return = data_to_return[['time','message', 'city','link','score','phone_number']]
        return data_to_return
    #final_data_to_reply_dataframe = pd.read_csv(final_data_to_reply)
    #final_data_to_reply_dataframe.sort_values(by=['tweet_id_requester', 'score'], ascending=False,inplace=True)



#tweets_extracted = tweet_processing(tweet_extraction(api))
