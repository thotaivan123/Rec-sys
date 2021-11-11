import pandas
import string
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
import re
import numpy as np
from pandas.core.frame import DataFrame
import streamlit as st
import streamlit.components.v1 as stc
import pandas as pd

import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

zomato_set = pd.read_csv("zomato (1).csv")
zomato_set.head()

# Deleting Unnnecessary Columns
zomato = zomato_set.drop(['url', 'dish_liked', 'phone'], axis=1)
# Dropping the column "dish_liked", "phone", "url"

# Removing the Duplicates
zomato.duplicated().sum()
zomato.drop_duplicates(inplace=True)

# Remove the NaN values from the dataset
zomato.isnull().sum()
zomato.dropna(how='any', inplace=True)

# Changing the column names
zomato = zomato.rename(columns={'approx_cost(for two people)': 'cost',
                       'listed_in(type)': 'type', 'listed_in(city)': 'city'})

zomato['cost'] = zomato['cost'].astype(str)  # Changing the cost to string
# Using lambda function to replace ',' from cost
zomato['cost'] = zomato['cost'].apply(lambda x: x.replace(',', '.'))
zomato['cost'] = zomato['cost'].astype(float)
# Removing '/5' from Rates
zomato = zomato.loc[zomato.rate != 'NEW']
zomato = zomato.loc[zomato.rate != '-'].reset_index(drop=True)
def remove_slash(x): return x.replace('/5', '') if type(x) == np.str else x


zomato.rate = zomato.rate.apply(remove_slash).str.strip().astype('float')

# Adjust the column names
zomato.name = zomato.name.apply(lambda x: x.title())
zomato.online_order.replace(('Yes', 'No'), (True, False), inplace=True)
zomato.book_table.replace(('Yes', 'No'), (True, False), inplace=True)

# Computing Mean Rating
restaurants = list(zomato['name'].unique())
zomato['Mean Rating'] = 0

for i in range(len(restaurants)):
    zomato['Mean Rating'][zomato['name'] == restaurants[i]
                          ] = zomato['rate'][zomato['name'] == restaurants[i]].mean()

scaler = MinMaxScaler(feature_range=(1, 5))
zomato[['Mean Rating']] = scaler.fit_transform(
    zomato[['Mean Rating']]).round(2)


zomato["reviews_list"] = zomato["reviews_list"].str.lower()


def remove_punctuation(text):
    # function to remove punctuation
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))


PUNCT_TO_REMOVE = string.punctuation

zomato["reviews_list"] = zomato["reviews_list"].apply(
    lambda text: remove_punctuation(text))


def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)


zomato["reviews_list"] = zomato["reviews_list"].apply(
    lambda text: remove_urls(text))

zomato[['reviews_list', 'cuisines']].sample(5)

restaurant_names = list(zomato['name'].unique())


def get_top_words(column, top_nu_of_words, nu_of_word):
    vec = CountVectorizer(ngram_range=nu_of_word, stop_words='english')
    bag_of_words = vec.fit_transform(column)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx])
                  for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:top_nu_of_words]


zomato = zomato.drop(['address', 'rest_type', 'type',
                     'menu_item', 'votes'], axis=1)

df_percent1 = zomato.sample(frac=0.5)

df_percent1.set_index('name', inplace=True)
indices = pd.Series(df_percent1.index)

df_percent1.shape

tfidf1 = TfidfVectorizer(analyzer='word', ngram_range=(
    1, 2), min_df=0, stop_words='english')
tfidf_matrix1 = tfidf1.fit_transform(df_percent1['reviews_list'])

cosine_similarities1 = linear_kernel(tfidf_matrix1, tfidf_matrix1)


def recommend(name, cosine_similarities=cosine_similarities1):

    # Create a list to put top restaurants
    recommend_restaurant = []

    # Find the index of the hotel entered
    idx = indices[indices == name].index[0]

    # Find the restaurants with a similar cosine-sim value and order them from bigges number
    score_series = pd.Series(
        cosine_similarities1[idx]).sort_values(ascending=False)

    top30_indexes = list(score_series.iloc[0:31].index)

    top_30li = pd.DataFrame(
        {
            'index': top30_indexes,
            'scores': top30_indexes
        }
    )

    z = []
    for ind in top_30li.index:
        if(top_30li['scores'][ind] > 0.2):
            y = top_30li['index'][ind]
            z.append(y)

    for each in z:
        recommend_restaurant.append(list(df_percent1.index)[each])

    df_new = pd.DataFrame(columns=['cuisines', 'Mean Rating', 'cost', ])

    for each in recommend_restaurant:
        df_new = df_new.append(pd.DataFrame(
            df_percent1[['cuisines', 'Mean Rating', 'cost']][df_percent1.index == each].sample()))

    df_new = df_new.drop_duplicates(
        subset=['cuisines', 'Mean Rating', 'cost'], keep=False)
    df_new = df_new.sort_values(by='Mean Rating', ascending=False).head(10)

    return df_new


recommend('Ice Land')

RESULT_TEMP = """
<div style="width:90%;height:100%;margin:1px;padding:5px;position:relative;border-radius:5px;border-bottom-right-radius: 60px;
box-shadow:0 0 15px 5px #ccc; background-color: #a8f0c6;
  border-left: 5px solid #6c6c6c;">
<h4>{}</h4>
<p style="color:blue;"><span style="color:black;">📈Score::</span>{}</p>
<p style="color:blue;"><span style="color:black;">🔗</span><a href="{}",target="_blank">Link</a></p>
<p style="color:blue;"><span style="color:black;">💲Price:</span>{}</p>
<p style="color:blue;"><span style="color:black;">🧑‍🎓👨🏽‍🎓 Students:</span>{}</p>

</div>
"""


def main():

    st.title("Restaurant Recommendation App")
    menu = ["Restaurants", "Recommendations"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Restaurants":
        st.subheader("Restaurants")
        st.write("courses data")
        st.dataframe(zomato_set.head(10))

    else:
        st.subheader("Recommendations")

        search_term = st.text_input("Search")

        num_of_rec = st.sidebar.number_input("Number", 4, 30, 7)
        if st.button("Recommend"):
            if search_term is not None:
                try:
                    results = recommend(search_term)
                    with st.expander("Results as JSON"):
                        results_json = results.to_dict('index')
                        st.write(results_json)
                    st.DataFrame(results)
                except:
                    results = "Not Found"
                    st.warning(results)


if __name__ == '__main__':
    main()