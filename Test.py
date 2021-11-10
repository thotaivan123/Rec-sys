import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
import re
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

zomato_set=pd.read_csv("zomato.csv")
zomato_set.head()

#Deleting Unnnecessary Columns
zomato=zomato_set.drop(['url','dish_liked','phone'],axis=1) 
#Dropping the column "dish_liked", "phone", "url"

#Removing the Duplicates
zomato.duplicated().sum()
zomato.drop_duplicates(inplace=True)

#Remove the NaN values from the dataset
zomato.isnull().sum()
zomato.dropna(how='any',inplace=True)

#Changing the column names
zomato = zomato.rename(columns={'approx_cost(for two people)':'cost','listed_in(type)':'type', 'listed_in(city)':'city'})

zomato['cost'] = zomato['cost'].astype(str) #Changing the cost to string
zomato['cost'] = zomato['cost'].apply(lambda x: x.replace(',','.')) #Using lambda function to replace ',' from cost
zomato['cost'] = zomato['cost'].astype(float)
#Removing '/5' from Rates
zomato = zomato.loc[zomato.rate !='NEW']
zomato = zomato.loc[zomato.rate !='-'].reset_index(drop=True)
remove_slash = lambda x: x.replace('/5', '') if type(x) == np.str else x
zomato.rate = zomato.rate.apply(remove_slash).str.strip().astype('float')

# Adjust the column names
zomato.name = zomato.name.apply(lambda x:x.title())
zomato.online_order.replace(('Yes','No'),(True, False),inplace=True)
zomato.book_table.replace(('Yes','No'),(True, False),inplace=True)

## Computing Mean Rating
restaurants = list(zomato['name'].unique())
zomato['Mean Rating'] = 0

for i in range(len(restaurants)):
    zomato['Mean Rating'][zomato['name'] == restaurants[i]] = zomato['rate'][zomato['name'] == restaurants[i]].mean()
    
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (1,5))
zomato[['Mean Rating']] = scaler.fit_transform(zomato[['Mean Rating']]).round(2)


zomato["reviews_list"] = zomato["reviews_list"].str.lower()

import string

def remove_punctuation(text):
    # function to remove punctuation
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

PUNCT_TO_REMOVE = string.punctuation

zomato["reviews_list"] = zomato["reviews_list"].apply(lambda text: remove_punctuation(text))

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

zomato["reviews_list"] = zomato["reviews_list"].apply(lambda text: remove_urls(text))

zomato[['reviews_list', 'cuisines']].sample(5)

restaurant_names = list(zomato['name'].unique())
def get_top_words(column, top_nu_of_words, nu_of_word):
    vec = CountVectorizer(ngram_range= nu_of_word, stop_words='english')
    bag_of_words = vec.fit_transform(column)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:top_nu_of_words]
    
zomato=zomato.drop(['address','rest_type', 'type', 'menu_item', 'votes'],axis=1)
import pandas

df_percent1 = zomato.sample(frac=0.5)

df_percent1.set_index('name', inplace=True)
indices = pd.Series(df_percent1.index)

df_percent1.shape

tfidf1 = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
tfidf_matrix1 = tfidf1.fit_transform(df_percent1['reviews_list'])

cosine_similarities1 = linear_kernel(tfidf_matrix1, tfidf_matrix1)

def recommend(name, cosine_similarities = cosine_similarities1):
    
    # Create a list to put top restaurants
    recommend_restaurant = []
    
    # Find the index of the hotel entered
    idx = indices[indices == name].index[0]
    
    # Find the restaurants with a similar cosine-sim value and order them from bigges number
    score_series = pd.Series(cosine_similarities1[idx]).sort_values(ascending=False)

    top30_indexes = list(score_series.iloc[0:31].index)

    top_30li = pd.DataFrame(
        {
            'index':top30_indexes,
            'scores':top30
        }
    )

    z = []
    for ind in top_30li.index:
      if(top_30li['scores'][ind]>0.2):
        y = top_30li['index'][ind]
        z.append(y)
 
    for each in z:
        recommend_restaurant.append(list(df_percent1.index)[each])
  
    df_new = pd.DataFrame(columns=['cuisines', 'Mean Rating', 'cost',])

    for each in recommend_restaurant:
        df_new = df_new.append(pd.DataFrame(df_percent1[['cuisines','Mean Rating', 'cost']][df_percent1.index == each].sample()))

    df_new = df_new.drop_duplicates(subset=['cuisines','Mean Rating', 'cost'], keep=False)
    df_new = df_new.sort_values(by='Mean Rating', ascending=False).head(10)
    
    print('TOP %s RESTAURANTS LIKE %s WITH SIMILAR REVIEWS: ' % (str(len(df_new)), name))
    
    return df_new

recommend('Ice Land')
