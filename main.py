# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np  # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df = pd.read_csv("/kaggle/input/sms-spam-collection-dataset/spam.csv", encoding='latin-1')


df.rename(columns={"v1": "target", "v2": "text"}, inplace=True)
df['target']


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()


df["target"] = encoder.fit_transform(df["target"])


df.drop_duplicates(keep='first')


df.head()


import matplotlib.pyplot as plt


import nltk

df['num_characters'] = df['text'].apply(len)
df['num_words'] = df['text'].apply(lambda x: len(nltk.word_tokenize(x)))
df['num_sentences'] = df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))
df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)

df.head()


df[df['target'] == 0][["num_characters", "num_words", "num_sentences"]].describe()

df[df['target'] == 1][["num_characters", "num_words", "num_sentences"]].describe()

import seaborn as sns

sns.histplot(df[df['target'] == 0]['num_characters'])
sns.histplot(df[df['target'] == 1]['num_characters'], color='red')

sns.heatmap(df.corr(),annot=True)

from nltk.tokenize import word_tokenize

df['text'] = df['text'].apply(lambda x: x.lower())


df['text'] = df['text'].apply(lambda x: word_tokenize(x))

df



def alnum(text):
    k = []
    for i in text:
        if i.isalnum():
            k.append(i)
    return k



# remove symbols
df['text'] = df['text'].apply(lambda x: alnum(x))


from nltk.corpus import stopwords


# remove stopwords

stop_words = set(stopwords.words('english'))


def stopwords(text):
    k = []
    for i in text:
        if i not in stop_words:
            k.append(i)
    return k


df['text'] = df['text'].apply(lambda x: stopwords(x))


import string


def rem_pun(x):
    x = [''.join(c for c in s if c not in string.punctuation) for s in x]
    return x


df['text'] = df['text'].apply(lambda x: rem_pun(x))

df.head()


from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def stemm(x):
    y = []
    for i in x:
        y.append(ps.stem(i))
    return y


df['text'] = df['text'].apply(lambda x: stemm(x))

df.head()

def listToString(s):
    # initialize an empty string
    str1 = " "

    # return string
    return (str1.join(s))


df["text"] = df["text"].apply(lambda x: listToString(x))

df


wc = WordCloud(width=500, height=500, min_font_size=10, background_color="white")

spam_words = wc.generate(df[df['target'] == 1]['text'].str.cat(sep=" "))

plt.figure
plt.imshow(spam_words)

_words = wc.generate(df[df['target'] == 0]['text'].str.cat(sep=" "))

plt.imshow(_words)

df


df["text"][0]

spam_corpus = []
for msg in df[df['target'] == 1]['text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)

len(spam_corpus)

from collections import Counter

sns.barplot(pd.DataFrame(Counter(spam_corpus).most_common(30))[0],
            pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')

c=[]
for msg in df[df['target'] == 0]['text'].tolist():
    for word in msg.split():
        c.append(word)

len(c)


sns.barplot(pd.DataFrame(Counter(c).most_common(30))[0], pd.DataFrame(Counter(c).most_common(30))[1])
plt.xticks(rotation='vertical')

# machine learning model---textual data works best with naive bayes algorithm


from sklearn.feature_extraction.text import CountVectorizer

# TfidVectorizer can be used
cv = CountVectorizer()


x = cv.fit_transform(df['text']).toarray()
x.shape
y = df['target'].values


from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)


from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score


gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()


y_pred1 = gnb.predict(x_test)
print(accuracy_score(y_test, y_pred1))
print(confusion_matrix(y_test, y_pred1))
print(precision_score(y_test, y_pred1))


mnb.fit(x_train, y_train)
y_pred2 = mnb.predict(x_test)
print(accuracy_score(y_test, y_pred2))
print(confusion_matrix(y_test, y_pred2))
print(precision_score(y_test, y_pred2))


bnb.fit(x_train, y_train)
y_pred3 = bnb.predict(x_test)
print(accuracy_score(y_test, y_pred3))
print(confusion_matrix(y_test, y_pred3))
print(precision_score(y_test, y_pred3))


# votingclassifier, stacking