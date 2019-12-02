import re,string,nltk,csv
import gensim
from gensim import corpora
from nltk.tokenize import word_tokenize , TweetTokenizer
from nltk.stem import WordNetLemmatizer
from gensim.models import LdaModel, LdaMulticore
# nltk.download('stopwords')
# nltk.download('wordnet')
from nltk.corpus import stopwords

with open ('/Users/keshavgaddhyan/Desktop/NLP project/trump_tweets.csv', 'r') as file:
    reader=csv.reader(file)
    tweets=list(reader)

stop_words = stopwords.words('english') + ['...']
punctuation=set(string.punctuation)
wnl=WordNetLemmatizer()
tknzr = TweetTokenizer()

trade_tweets=["trade", "currency", "economy", "growth",  "trade-war", "rates", "inflation", "manipulation", "dollar", "Fed", "Powell", "tariffs", ]
for i in range(len(trade_tweets)):
    trade_tweets[i]=wnl.lemmatize(trade_tweets[i])  #lemmatize the words in our bag of words for trade realted tweets as well so that it is coherant with our lemmatized tweeets

for i in range (len(tweets)):
    tweets[i][0]=tweets[i][0].lower()                    # converts the tweets tro lower case
    tweets[i][0] = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', tweets[i][0]) # removes links in the tweets
    tweets[i][0]=tknzr.tokenize(tweets[i][0])                   #tokenize the tweets into words

for i in range (len(tweets)):
    for j in range(len(tweets[i][0])):                       #lemmatize the words
        tweets[i][0][j]=wnl.lemmatize(tweets[i][0][j])


for i in range (len(tweets)):
    for word in tweets[i][0]:
        if word in (stop_words):                             # remove stop words from the tweets
            tweets[i][0].remove(word)
        if word in punctuation:                              # remove punctuations from the tweets
            tweets[i][0].remove(word)

# print(tweets[11][0])
new_tweets=[]
for i in range(len(tweets)):
    new_tweets.append(tweets[i][0])


# print(type(new_tweets[0]))

dictionary = corpora.Dictionary(new_tweets)
# print(dictionary)

# mycorpus is a list of list
# where each list is a tweet and it is a list of tuples [(token_id, no_of_times_it_occurs_in_tweet),(),...]
mycorpus = [dictionary.doc2bow(tweet, allow_update=True) for tweet in new_tweets]

word_counts = [[(dictionary[id], count) for id, count in line] for line in mycorpus]

# lda_model = LdaModel(corpus=mycorpus,id2word=dictionary, num_topics=30,passes=10)
# lda_model = gensim.models.ldamodel.LdaModel(corpus=mycorpus,id2word=dictionary,
# num_topics=30, passes = 10)
# for idx, topic in lda_model.print_topics(-1):
#     print('Topic: {} \nWords: {}'.format(idx, topic))


trading_tweets=[]
for i in range (len(tweets)):
    for word in trade_tweets:           
        if word in tweets[i][0]:
            trading_tweets.append(tweets[i][0])
            break;


print(len(trading_tweets))
# print(trading_tweets[50])



def analyze_sentiment(tweet):
    analysis = TextBlob(tweet)
    if analysis.sentiment.polarity > 0.7:
        return 'Positive'
    elif analysis.sentiment.polarity < -0.7:
        return 'Negative'
    else:
        return 'Neutral'
# now dictionary and mycorpus will be inputs to LDA
# print(new_tweets[0])
# print(new_tweets[1])
# print(type(dictionary.token2id))
# print(dictionary.token2id.get('text'))
# print(mycorpus[1])
# print(len(mycorpus[1]))
# print(word_counts[1])
# print(len(word_counts[1]))
# print(punctuation)
# print(stop_words)


