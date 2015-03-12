import pandas as pd
import graphlab as gl
from random import random
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize


# ##Feature Engineering
# This iteration will take into account several simple features:
# activity, loan_amount, country, posted_date, sector


# clean pairs data
def clean_pair_date(sf):
    sf.rename({'X1': 'lender_id', 'X2': 'loan_id'})
    sf['loan_id'] = sf['loan_id'].astype(str)
    sf['lender_id'] = sf['lender_id'].astype(str)
    return sf


# clean loan data
def clean_loan_data(df):
    # drop columns
    df = df.drop(['paid_date', 'planned_expiration_date', 'languages'], axis=1)

    # drop duplicates
    df = df.drop_duplicates('id')

    # drop duplicate header
    df = df[df['activity'] != 'activity']

    # drop nas
    df = df.dropna(subset=['earliest_scheduled_payment', 'last_scheduled_payment',
                           'repayment_interval', 'posted_date',
                           'status', 'repayment_term', 'use'], how='any')

    # fill paid_amount's na with zero
    df['paid_amount'] = df['paid_amount'].fillna(0)

    # fill genders
    df['gender'] = df['gender'].map(lambda x: 'M' if random() <= 0.39 else 'F')
    df['gender'] = df['gender'].map(lambda x: 1 if x == 'F' else 0)

    # fill null descriptions with empty string
    df['descriptions'] = df['descriptions'].fillna(0)

    # binaralize bonus credit
    df['bonus_credit_eligibility'] = df['bonus_credit_eligibility'].map(lambda x: 1 if x == 'True' else 0)

    # convert some columns to datetime
    df['earliest_scheduled_payment'] = df['earliest_scheduled_payment'].map(
        lambda x: pd.to_datetime(x, format='%Y-%m-%dT%H:%M:%SZ'))
    df['last_scheduled_payment'] = df['last_scheduled_payment'].map(
        lambda x: pd.to_datetime(x, format='%Y-%m-%dT%H:%M:%SZ'))
    df['posted_date'] = df['posted_date'].map(
        lambda x: pd.to_datetime(x, format='%Y-%m-%dT%H:%M:%SZ'))

    # convert some columns to int
    df['lender_count'] = df['lender_count'].astype('int64')
    df['loan_amount'] = df['loan_amount'].astype('int64')
    df['paid_amount'] = df['paid_amount'].astype('float64').astype('int64')
    df['repayment_term'] = df['repayment_term'].astype('float64')

    return df


def drop_unexsiting_loan_ids(sf, df):
    loan_ids_in_pairs = sorted(list(sf['loan_id'].unique()))
    loan_ids_in_loans = sorted(list(df['id'].values))
    loan_ids_intersection = set(loan_ids_in_loans) & set(loan_ids_in_pairs)
    # drop useless loan_ids in sf
    sf['loan_id'] = sf['loan_id'].apply(lambda x: x if x in loan_ids_intersection else None)
    sf = sf.dropna('loan_id')
    # drop useless loan_ids in df
    df['id'] = df['id'].map(lambda x: x if x in loan_ids_intersection else None)
    df = df.dropna()
    return sf, df


# dummify posted_date to seasons
def convert_to_season(x):
    m = x.month
    if m < 4:
        return 'Spring'
    elif m < 7:
        return 'Summer'
    elif m < 11:
        return 'Fall'
    else:
        return 'Winter'


def tokenize(doc):
    '''
    INPUT: string
    OUTPUT: list of strings

    Tokenize and stem the document.
    '''
    snowball = SnowballStemmer('english')
    return [snowball.stem(word) for word in word_tokenize(doc.lower())]


def get_vectorizer(descriptions, num_features=100):
    vect = TfidfVectorizer(max_features=num_features, stop_words='english', tokenizer=tokenize)
    return vect.fit(descriptions)


# getting features
def get_loan_features(df):
    raw_features = df[['bonus_credit_eligibility', 'loan_amount', 'posted_date',
                       'use', 'gender', 'family', 'country', 'repayment_term',
                       'repayment_interval', 'id']]

    raw_features['season'] = raw_features['posted_date'].map(lambda x: convert_to_season(x))
    raw_features = pd.concat([raw_features,
                              pd.get_dummies(raw_features['season'], prefix='season_')], axis=1)
    raw_features = raw_features.drop(['season', 'posted_date'], axis=1)

    # dummify repayment_interval
    raw_features = pd.concat([raw_features,
                              pd.get_dummies(raw_features['repayment_interval'],
                              prefix='repayment_interval_')], axis=1)
    raw_features = raw_features.drop(['repayment_interval'], axis=1)

    # get tfidf of 'use'
    text = raw_features['use'].values
    tfidf = pd.DataFrame(get_vectorizer(text).transform(text).toarray())
    tfidf.columns = tfidf.columns.astype(str)
    tfidf = tfidf.astype(float)
    raw_features = pd.concat([raw_features, tfidf], axis=1, join_axes=[raw_features.index])
    # drop use
    raw_features = raw_features.drop(['use'], axis=1)
    raw_features = raw_features.fillna(0)
    loan_feature = gl.SFrame(raw_features.to_dict(orient='list'))
    loan_feature.rename({'id': 'loan_id'})
    return loan_feature


# run the model
def run_model(sf, df, loan_feature):
    # split train test
    train, test = gl.recommender.util.random_split_by_user(sf, user_id='lender_id', item_id='loan_id', item_test_proportion=0.3)

    # compare models
    m = gl.recommender.ranking_factorization_recommender.create(train,
                                                                user_id='lender_id',
                                                                item_id='loan_id',
                                                                item_data=loan_feature,
                                                                num_factors=5,
                                                                regularization=0.01,
                                                                binary_target=True,
                                                                max_iterations=20,
                                                                verbose=True)

    print m.evaluate(test, metric='precision_recall')
    print '='*10, 'Summary', '='*10
    print m.summary()
    m.save('models/iter5_pipeline')


if __name__ == '__main__':
    # load pairs into SFrame
    sf = gl.SFrame.read_csv('data/lender_loan_pairs.csv', header=False, delimiter=',', verbose=False)
    sf = clean_pair_date(sf)

    # Create side features

    df = pd.read_csv('data/loans.csv', delimiter=',')
    df = clean_loan_data(df)
    sf, df = drop_unexsiting_loan_ids(sf, df)

    loan_feature = get_loan_features(df)

    run_model(sf, df, loan_feature)
