import numpy as np
from sklearn.impute import IterativeImputer
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from collections import Counter
from pylab import *
from bs4 import BeautifulSoup
import unicodedata
from contractions import contractions_dict
from nltk.tokenize import RegexpTokenizer
import re
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout
from keras.layers import LSTM
from keras.activations import sigmoid

mich = pd.read_csv('../../Downloads/MIchigan.csv', on_bad_lines='skip')
louis = pd.read_csv('../../Downloads/Louisville.csv', on_bad_lines='skip')
alex = pd.read_csv('../../Downloads/Alexandria.csv', on_bad_lines='skip')
exp1 = pd.read_csv('../../Downloads/michigan_2.csv', on_bad_lines='skip')
exp2 = pd.read_csv('../../Downloads/Kentucky_2500.csv', on_bad_lines='skip')
exp3 = pd.read_csv('../../Downloads/Parse_23Nov.csv', on_bad_lines='skip')
low1 = pd.read_csv('../../Downloads/low_1.csv', on_bad_lines='skip')
df = pd.concat([mich, louis, alex, exp1, exp2, exp3, low1])


# removes html tags
def remove_html(text):
    CLEANR = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    cleantext = re.sub(CLEANR, '', text)
    return cleantext


# tokenizing
def tknise(text):
    tkn = nltk.word_tokenize(text)
    return tkn


# lemmatizatioan and converting to lowercase
def lem_and_lower(text_tokens):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmatized_tokens = [(lemmatizer.lemmatize(token)).lower() for token in
                         text_tokens if token.isalpha()]
    return lemmatized_tokens


# Removing punctuation and stopwords
def rem_stopwords_punc(text_tokens):
    tokenizer = RegexpTokenizer(r'\w+')
    filtered = [tokenizer.tokenize(token)[0] for token in text_tokens if not token in
                                                                             stopwords.words('english') if
                token.isalpha()]
    return filtered


# covertinglisttostring
def listToString(s):
    str1 = " "
    return (str1.join(s))


def indexing(tokenized):
    words = [j for i in tokenized for j in i]
    count_words = Counter(words).most_common()
    vocab_index = {w: i + 1 for i, (w, c) in enumerate(count_words)}
    index_encoded1 = []
    for i in tokenized:
        r = [vocab_index[w] for w in i]
        index_encoded1.append(r)
    index_encoder = LabelEncoder()
    index_encoder = index_encoder.fit(words)
    index_encoded2 = [index_encoder.transform(doc) for doc in tokenized]
    index_encoded2l = [index_encoder.transform(doc).tolist() for doc in tokenized]
    return [index_encoded2, index_encoded2l]


def one_hot_encoding(tokenized):
    words = [j for i in tokenized for j in i]
    words_list = [[i] for i in words]
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoder = onehot_encoder.fit(words_list)
    onehot_encoded3 = [onehot_encoder.transform([[word] for word in doc]) for doc in tokenized]
    return (onehot_encoded3)


# pos tagging

def postag(string):
    data = [string]
    POS_c2 = []
    for doc in data:
        token_doc = nltk.word_tokenize(doc)
        POS_token_doc = nltk.pos_tag(token_doc)
        POS_token_temp = []
        for i in POS_token_doc:
            POS_token_temp.append(i[0] + i[1])
        POS_c2.append(" ".join(POS_token_temp))
    return POS_c2[0]


def StringTolist(text):
    return text.split(' ')


def clean(a):
    return a.replace("QR Code Link to This Post", "").replace("- ", "").replace(
        "ft2", "").replace('qr', "").replace('code', '').replace(
        'st', '').replace('w', '').strip()


# Removing HTML tags
def strip_html_tags(text):
    soup = BeautifulSoup(text, 'html.parser')
    stripped_text = soup.get_text()
    return stripped_text


# Remove accented characters
def remove_accents(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text


# Expanding contractions
def expand_contractions(text, contraction_mapping=contractions_dict):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) \
            if contraction_mapping.get(match) \
            else contraction_mapping.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text


# Tokenize and remove stopwords

excluding = ['against', 'not', 'don', "don't", 'ain', 'aren', "aren't", 'couldn', "couldn't",
             'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
             'haven', "haven't", 'isn', "isn't", 'mightn', "mightn't", 'mustn', "mustn't",
             'needn', "needn't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren',
             "weren't", 'won', "won't", 'wouldn', "wouldn't"]

stopwords_list = [word for word in stopwords if word not in excluding]


def remove_stopwords(text):
    tokens = nltk.word_tokenize(text)
    lemmatizer = nltk.stem.WordNetLemmatizer()
    tokens = [token.strip() for token in tokens]
    tokens = [(lemmatizer.lemmatize(token)).lower() for token in tokens if token.isalpha()]
    filtered_tokens = [token for token in tokens if token not in stopwords_list]
    return filtered_tokens


def processed_text(text):
    # Remove HTML tags
    processed = strip_html_tags(text)
    # Remove accents
    processed = remove_accents(processed)
    # Convert to lowercase
    processed = processed.lower()
    # Remove extra newlines
    processed = re.sub(r'[\r|\n|\r\n]+', ' ', processed)
    # Remove extra whitespace
    # # Expand contractions
    # processed = expand_contractions (processed)
    # Lemmatize, lowercase and remove stopwords
    processed = remove_stopwords(processed)
    return processed


def remove_punc(test_str):
    res = re.sub(r'[^\w\s]', '', test_str)
    return res


def remove_common_uncommon(a):
    rem_words = ['home', 'contact', 'call', 'info', 'http', 'need', 'see', 'u', 'guys',
                 'qiming', 'stdear', 'tx', 'tired', 'peep']
    s = ''
    for i in a.split(' '):
        if i in rem_words or len(a) < 3:
            i = ''
        s = s + ' ' + i
    return s


# cleaning and multiple imputation
def Clean_Impute(df):
    n_br_list = df.n_br.values.tolist()
    Area_list = df.Area.values.tolist()
    Area_list_num = [""] * len(n_br_list)

    # cleaning Number Bedroom Coloumn
    for index, value in enumerate(n_br_list):
        if len(n_br_list[index]) > 1:
            Area_list[index] = n_br_list[index][0:3]
            n_br_list[index] = n_br_list[index][len(n_br_list[index]) - 1]

    # cleaning Area Coloumn
    for index, value in enumerate(Area_list):
        if Area_list[index] != "":
            Area_list_num[index] = int(Area_list[index])
        else:
            Area_list_num[index] = ""
    n_br_list_num = [eval(i) for i in n_br_list]
    df["n_br"] = n_br_list_num
    df["Area"] = Area_list_num

    # replacing blank values with NaN
    df.replace('', np.nan, inplace=True)

    # imputation
    imputer = IterativeImputer(random_state=42)
    imputed = imputer.fit_transform(df[['n_br', 'Area']])
    print(len(imputed[0]))

    df["n_br"] = imputed[:, 0]
    df["Area"] = imputed[:, 1]


# cleaning the dataset
df['text4'] = df['Text2'] + df['Text3']
df['text4'] = df['text4'].astype(str)
df['Description'] = df['text4'].apply(remove_html)
df['tokenized'] = df['Description'].apply(tknise)
df['lem_lower'] = df['tokenized'].apply(lem_and_lower)
df['filtered_string'] = df['lem_lower'].apply(listToString)
df['processed'] = df['filtered_string'].apply(clean)
df['processed'] = df['processed'].apply(remove_punc)
df['processed'] = df['processed'].apply(remove_common_uncommon)
df['processed'] = df['processed'].apply(remove_stopwords)

# dropping null values
df.dropna(inplace=True)

# converting orice to float dtype
l = []
for i in df['Text'].values:
    for j in i:
        if not j.isdigit():
            i = i.replace(j, '')
    l.append(int(i))
df['price'] = l

# converting area and n_bedrooms to numeric type
df[['n_br', 'Area']] = mich.Text1.str.split(n=1, pat=' - ', expand=True)
df = df.dropna(axis=0)
df.drop(['Text1'], axis=1, inplace=True)
df['n_br'] = df['n_br'].str.replace("/ ", "")
df['n_br'] = df['n_br'].str.replace("br", "")
df['n_br'] = df['n_br'].str.strip()
df['Area'] = df['Area'].str.replace("- ", "")
df['Area'] = df['Area'].str.replace("ft2", "")
df['Area'] = df['Area'].str.strip()

df_for_model = df[['processed', 'price', 'n_br', 'Area']]

df_for_model.reset_index(inplace=True)

# filtering out best range after multiple attempts
df_for_model = df_for_model[(df_for_model['price'] > 500) & (df_for_model['price'] < 15000)]

# data partitioning
X_train, X_test, y_train, y_test = train_test_split(df_for_model['processed'], df_for_model['price'], test_size=0.2,
                                                    random_state=87)

vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=2)
X_train_trans = vectorizer.fit_transform(X_train)

# Using gridsearch to tune hyperparameters
sgd = SGDRegressor()
params = {
    'max_iter': [10000, 100000],
    'penalty': ['l2', 'l1', 'elasticnet']
}
grid_search = GridSearchCV(estimator=sgd,
                           param_grid=params,
                           cv=4,
                           n_jobs=-1, verbose=1, scoring='r2', error_score='raise')
grid_search.fit(X_train_trans, y_train)
best_sgd = grid_search.best_estimator_
best_sgd.fit(X_train_trans, y_train)
y_predicted_sgd_train = best_sgd.predict(X_train_trans)
train_sgd_rsqured = r2_score(y_train, y_predicted_sgd_train)
mape_train_sgd = mean_absolute_percentage_error(y_train, y_predicted_sgd_train)
y_test_predicted_sgd = best_sgd.predict(vectorizer.transform(X_test))
mape_test_sgd = mean_absolute_percentage_error(y_test, y_test_predicted_sgd)

rf = RandomForestRegressor(random_state=42, n_jobs=-1)
params = {
    'max_depth': [1000, 5000, 10000],
    'min_samples_leaf': [2, 5, 10, 20],
}
grid_search = GridSearchCV(estimator=rf,
                           param_grid=params,
                           cv=4,
                           n_jobs=-1, verbose=1, scoring='r2', error_score='raise')
grid_search.fit(X_train_trans, y_train)
best_rf = grid_search.best_estimator_
best_rf.fit(X_train_trans, y_train)
y_predicted_rf_train = best_rf.predict(X_train_trans)
train_rf_rsqured = r2_score(y_train, y_predicted_rf_train)
mape_train_rf = mean_absolute_percentage_error(y_train, y_predicted_rf_train)
y_test_predicted_rf = best_rf.predict(vectorizer.transform(X_test))
mape_test_rf = mean_absolute_percentage_error(y_test, y_test_predicted_rf)

xgb = xgb.XGBRegressor(n_jobs=-1)
params = {
    'n_estimators': [500, 1000, 5000],
    'learning_rate': [0.01, 0.1, 0.2, ],
    'max_depth': [5, 7, 9, 11]
}
grid_search = GridSearchCV(estimator=xgb,
                           param_grid=params,
                           cv=4,
                           n_jobs=-1, verbose=1, scoring='r2', error_score='raise')
grid_search.fit(X_train_trans, y_train)
best_xgb = grid_search.best_estimator_
best_xgb.fit(X_train_trans, y_train)
y_predicted_xgb_train = best_xgb.predict(X_train_trans)
train_xgb_rsqured = r2_score(y_train, y_predicted_xgb_train)
mape_train_xgb = mean_absolute_percentage_error(y_train, y_predicted_xgb_train)
y_test_predicted_xgb = best_xgb.predict(vectorizer.transform(X_test))
mape_test_xgb = mean_absolute_percentage_error(y_test, y_test_predicted_xgb)

svr = SVR(C=10000, epsilon=0.1)
params = {
    'C': [10000, 100000],
    'epsilon': [0.01, 0.1]
}
grid_search = GridSearchCV(estimator=svr,
                           param_grid=params,
                           cv=4,
                           n_jobs=-1, verbose=1, scoring='r2', error_score='raise')
grid_search.fit(X_train_trans, y_train)
best_svr = grid_search.best_estimator_
best_svr.fit(X_train_trans, y_train)
y_pred_train_svr = best_svr.predict(X_train_trans)
mape_svr_train = mean_absolute_percentage_error(y_train, y_pred_train_svr)
r2_svr = r2_score(y_train, y_pred_train_svr)
y_pred_test_svr = best_svr.predict(vectorizer.transform(X_test))
mape_svr_test = mean_absolute_percentage_error(y_test, vectorizer.transform(X_test))

# Trying to improve the model with adding more variables
features2 = ['n_br', 'Area']
X2 = df_for_model.loc[:, features2]
Y = df_for_model.loc[:, ['price']]
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, Y, test_size=0.2, random_state=87)
Clean_Impute(X2_train)
Clean_Impute(X2_test)
best_xgb.fit(X2_train, y2_train)
y2_pred_train = best_xgb.predict(X2_train)
score1 = r2_score(y2_train, y2_pred_train)
y2_pred_test = best_xgb.predict(X2_test)
score = mean_squared_error(y2_test, y2_pred_test, squared=False)
y_pred_ense_train = (y_predicted_xgb_train + y2_pred_train) / 2
score_xgb_2 = r2_score(y2_train, y_pred_ense_train)

# converting the problem into a classification problem and running an LSTM
df = pd.read_csv('../../Downloads/for_lstm.csv')
y = pd.get_dummies(df['price'])
X = df['processed']


def str_to_list(a):
    return a.split(' ')


def indexing(tokenized):
    words = [j for i in tokenized for j in i]
    count_words = Counter(words).most_common()
    vocab_index = {w: i + 1 for i, (w, c) in enumerate(count_words)}
    index_encoded1 = []
    for i in tokenized:
        r = [vocab_index[w] for w in i]
        index_encoded1.append(r)
    index_encoder = LabelEncoder()
    index_encoder = index_encoder.fit(words)
    index_encoded2 = [index_encoder.transform(doc) for doc in tokenized]
    return index_encoded2


def index_to_one_hot(index_encoded):
    indices = [j for i in index_encoded for j in i]
    indices_list2 = np.array(indices).reshape(-1, 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoder = onehot_encoder.fit(indices_list2)
    # encoding
    onehot_encoded2 = [onehot_encoder.transform(doc_i.reshape(-1, 1)).tolist() for doc_i in
                       np.array(index_encoded)]
    return np.array(onehot_encoded2)


df['processed'] = df['processed'].apply(str_to_list)
index = indexing(df['processed'])

# padding
seq = pad_sequences(index, maxlen=100, dtype='float64', padding='post', value=0.0)

# finding all the words to input number of features into LSTM
words = [j for i in df['processed'] for j in i]

X_train, X_test, y_train, y_test = train_test_split(seq, y, test_size=0.25, random_state=4)

# Architecting an LSTM model
LSTMmodel = Sequential()
LSTMmodel.add(Embedding(len(words), 50))
LSTMmodel.add(LSTM(40, dropout=0.2, recurrent_dropout=0.2))
LSTMmodel.add(Dropout(0.2))
LSTMmodel.add(Dense(5, activation='sigmoid'))
LSTMmodel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = LSTMmodel.fit(X_train, y_train, batch_size=120, epochs=50, validation_split=0.15)
performance = LSTMmodel.evaluate(X_test, y_test)
test_accuracy = performance[1]
