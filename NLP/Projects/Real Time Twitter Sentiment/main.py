API_key="rLlNmh0d7YQSXyUzGnS2ZT3dP"
API_key_secret="mQPKnzIVVDDstne4ATBtHrc8AyA1pP2UnHMaYVQJ60zOFb8tTW"
Access_token="1327939945307066376-oAVgx4YVYkxETAxBwYyQgH8QKkZWcp"
Access_token_secret="FFtj562IEMEQTz9nAHUE68eWPgvz9thKmdOrPlltuclwy"
Bearer_token="AAAAAAAAAAAAAAAAAAAAAMEYJwEAAAAA1Q8D%2BU%2FnEREHmSzKONiOiXfksu8%3DFMLanfCXhfT1Usd6b1yFfTO5brOUqo6iaoJkHYHtCD2tukEk59"

import tweepy
from tweepy import api

auth = tweepy.OAuthHandler(API_key,API_key_secret)
auth.set_access_token(Access_token,Access_token_secret)

api=tweepy.API(auth)

# Get My timeline details

# print(api.home_timeline())
twittes=api.home_timeline()

# for tweet in twittes:
#     print(tweet.text)
#polarity=Politness
#subjectivity=Personal Opinion
#objectivity=fact

### Get twitter stream and do sentiment analysis

from tweepy import Stream,streaming,StreamListener
import json
from textblob import TextBlob
import re
import csv
import datetime

Trump=0
Biden=0
header_name=['Trump','Biden','Time']
with open('sentiment.csv','w') as file:
    writer = csv.DictWriter(file,fieldnames=header_name)
    writer.writeheader()

    



class MyListener(StreamListener):
    def on_data(self, raw_data):
        raw_twittes=json.loads(raw_data)
        try:
            tweets=raw_twittes['text']
            # print(tweets)
            tweets = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweets).split())
            tweets = ' '.join(re.sub('RT',' ', tweets).split())
            blob=TextBlob(tweets.strip())

            global Trump
            global Biden

            Trump_sentiment=0
            Biden_sentiment=0

            for sent in blob.sentences:
                if "Trump" in sent and "Biden" not in sent:
                    Trump_sentiment=Trump_sentiment + sent.sentiment.polarity
                else:
                    Biden_sentiment=Biden_sentiment + sent.sentiment.polarity
                
            Trump=Trump+Trump_sentiment
            Biden=Biden+Biden_sentiment
            print(Trump)
            print(Biden)

            # with open('sentiment.csv','a') as file:
            #     writer=csv.DictWriter(file,fieldnames=header_name)
                
            #     info={
            #         'Trump':Trump,
            #         'Biden':Biden,
            #         'Time':now
            #     }
            #     writer.writerow(info)
            with open('sentiment.csv', 'a') as file:
                writer = csv.DictWriter(file, fieldnames=header_name)
                # now=datetime.now().strftime.tostr
                # print(now)
                info = {
                    'Trump': Trump,
                    'Biden': Biden,
                    # 'Time':now,
                }
                writer.writerow(info)
                # print(datetime.now)

            # print(blob)

        except:
            print('Error')
    
    # def on_status(self,status):
        # print(status)
    # def on_exception(self, exception):
        # print(exception.text)
    def on_error(self, status_code):
        print(status_code)
    
    
twitter_stream=Stream(auth,listener=MyListener())
twitter_stream.filter(track=['Trump','Biden'])

