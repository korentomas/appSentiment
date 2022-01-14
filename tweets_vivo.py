import tweepy as tw
import pandas as pd
import joblib
import streamlit as st
from st_aggrid import AgGrid
import regex as re
import string

consumer_key = 'w6J8TpaCuj0iJx3zl9QYGSSfn'
consumer_secret = '4GKAsQoIrBpWpRH1zKNmJVWfeAtNqJq0kLODO0SGKgR3CWpORw'
access_token = '501467554-gxZCTy4jOhiwleMOcM8YQdiDDO0pHVB9Otfbepah'
access_token_secret = 'ZllrVCsaVlThmXeAQsOu6NaDWNKg9mAjJFtRlURb7jqwg'
auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

classifier = joblib.load('sentiment-model.pkl')



##CUSTOM DEFINED FUNCTIONS TO CLEAN THE TWEETS

#Remove punctuations, links, mentions and \r\n new line characters
def strip_all_entities(text): 
    text = text.replace('\r', '').replace('\n', ' ').replace('\n', ' ').lower() #remove \n and \r and lowercase
    text = re.sub(r"(?:\@|https?\://)\S+", "", text) #remove links and mentions
    text = re.sub(r'[^\x00-\x7f]',r'', text) #remove non utf8/ascii characters such as '\x9a\x91\x97\x9a\x97'
    banned_list= string.punctuation + 'Ã'+'±'+'ã'+'¼'+'â'+'»'+'§'
    table = str.maketrans('', '', banned_list)
    text = text.translate(table)
    return text

#clean hashtags at the end of the sentence, and keep those in the middle of the sentence by removing just the # symbol
def clean_hashtags(tweet):
    new_tweet = " ".join(word.strip() for word in re.split('#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)', tweet)) #remove last hashtags
    new_tweet2 = " ".join(word.strip() for word in re.split('#|_', new_tweet)) #remove hashtags symbol from words in the middle of the sentence
    return new_tweet2

#Filter special characters such as & and $ present in some words
def filter_chars(a):
    sent = []
    for word in a.split(' '):
        if ('$' in word) | ('&' in word):
            sent.append('')
        else:
            sent.append(word)
    return ' '.join(sent)

def remove_mult_spaces(text): # remove multiple spaces
    return re.sub("\s\s+" , " ", text)

st.title('Sentiment Analysis G8 DigitalHouse')


st.markdown(
    'This app uses tweepy to get tweets from twitter based on the input name/phrase. It then processes the tweets through our model for sentiment analysis. The resulting sentiments and corresponding tweets are then put in a dataframe for display which is what you see as result.'
)


def run():
    with st.form(key='Enter name'):
        search_words = st.text_input(
            'Enter the name for which you want to know the sentiment')
        number_of_tweets = st.number_input(
            'Enter the number of latest tweets for which you want to know the sentiment(Maximum 50 tweets)',
            0, 50, 10)
        submit_button = st.form_submit_button(label='Submit')
        if submit_button:
            tweets = tw.Cursor(api.search_tweets, q=search_words,
                               lang='en').items(number_of_tweets)
            tweet_list = [i.text for i in tweets]

            texts_new = []
            for t in tweet_list:
                texts_new.append(remove_mult_spaces(filter_chars(clean_hashtags(strip_all_entities(t)))))

            df = pd.DataFrame(list(zip(texts_new, classifier.predict((texts_new)))),
                             columns=[
                                  'Latest ' + str(number_of_tweets) +
                                  ' Tweets' + ' on ' + search_words,
                                  ' Sentiment'
                             ]
                             )
            AgGrid(df, fit_columns_on_grid_load=True)

    
    st.markdown('This app analyses the sentiment behind a sentence.')

    with st.form(key='Type sentence'):
        sentence = st.text_input('Type a sentence for which you want to know the sentiment')
        submit_button_type = st.form_submit_button(label='Submit')
        
        if submit_button_type:

            result_new = []
            for t in [sentence]:
                result_new.append(remove_mult_spaces(filter_chars(clean_hashtags(strip_all_entities(t)))))


            result = classifier.predict((result_new))

            if result[0] == 'NEG':
                st.error('Negative')
            elif result[0] == 'POS':
                st.success('Positive')
            #st.write(result[0])

if __name__ == '__main__':
    run()