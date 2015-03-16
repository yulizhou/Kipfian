import pandas as pd
import graphlab as gl
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import StandardScaler
import string


PAIR_PATH = '../data/cleaned_lender_loan_pairs.csv'
LOAN_PATH = '../data/cleaned_loans.csv'
LENDER_PATH = '../data/cleaned_lender.csv'


# Feature Engineering

def tokenize(doc):
    '''
    INPUT: string
    OUTPUT: list of strings

    Tokenize and stem the document.
    '''
    snowball = SnowballStemmer('english')
    punc = set(string.punctuation)
    return [snowball.stem(word) for word in word_tokenize(doc.lower()) if word not in punc]


def get_vectorizer(descriptions, num_features=100):
    vect = TfidfVectorizer(max_features=num_features, stop_words='english', tokenizer=tokenize)
    return vect.fit(descriptions)


# getting features
def get_loan_features(df):
    df = df[['bonus_credit_eligibility', 'loan_amount', 'lender_count',
             'use', 'gender', 'family', 'country', 'repayment_term',
             'repayment_interval', 'id', 'posted_year', 'posted_month',
             'earliest_scheduled_payment', 'last_scheduled_payment']]

    # standardize lender_count
    df['lender_count'] = StandardScaler().fit_transform(df['lender_count'])

    # transform two datetime features into 'yearmonth' format
    df['earliest_scheduled_payment'] = df['earliest_scheduled_payment'].map(
        lambda x: pd.to_datetime(x, format='%Y-%m-%dT%H:%M:%SZ'))
    df['earliest_scheduled_payment_date'] = \
        df['earliest_scheduled_payment'].map(lambda x: str(x.year) + str(x.month))

    df['last_scheduled_payment'] = df['last_scheduled_payment'].map(
        lambda x: pd.to_datetime(x, format='%Y-%m-%dT%H:%M:%SZ'))
    df['last_scheduled_payment_date'] = \
        df['last_scheduled_payment'].map(lambda x: str(x.year) + str(x.month))

    df.drop(['earliest_scheduled_payment', 'last_scheduled_payment'],
            axis=1, inplace=True)

    # dummify repayment_interval
    df = pd.concat([df, pd.get_dummies(df['repayment_interval'],
                                       prefix='repayment_interval_')], axis=1)
    df = df.drop(['repayment_interval'], axis=1)

    # get tfidf of 'use'
    text = df['use'].values
    tfidf = pd.DataFrame(get_vectorizer(text).transform(text).toarray())
    tfidf.columns = tfidf.columns.astype(str)
    tfidf = tfidf.astype(float)
    df = pd.concat([df, tfidf], axis=1, join_axes=[df.index])

    # drop use
    df = df.drop(['use'], axis=1)
    df = df.fillna(0)
    loan_feature = gl.SFrame(df.to_dict(orient='list'))
    loan_feature.rename({'id': 'loan_id'})
    return loan_feature


def get_lender_features(df):
    df = df[['lender_id', 'loan_count', 'member_since']]

    df['loan_count'] = StandardScaler().fit_transform(df['loan_count'])

    df['member_since'] = \
        df['member_since'].map(lambda x: pd.to_datetime(x, format='%Y-%m-%dT%H:%M:%SZ'))
    df['mem_year'] = df['member_since'].map(lambda x: x.year).astype(str)
    df['mem_month'] = df['member_since'].map(lambda x: x.month).astype(str)
    df.drop(['member_since'], axis=1, inplace=True)

    return gl.SFrame(df.to_dict(orient='list'))


# run the model
def run_model(sf, loan_feature, lender_feature):
    # split train test
    train, test = gl.recommender.util.random_split_by_user(sf,
                                                           user_id='lender_id',
                                                           item_id='loan_id',
                                                           item_test_proportion=0.2)

    # compare models
    m = gl.recommender.ranking_factorization_recommender.create(train,
                                                                user_id='lender_id',
                                                                item_id='loan_id',
                                                                user_data=lender_feature,
                                                                item_data=loan_feature,
                                                                num_factors=20,
                                                                regularization=1e-9,
                                                                binary_target=False,
                                                                max_iterations=20,
                                                                num_sampled_negative_examples=1000,
                                                                verbose=True)

    print m.evaluate(test, metric='precision_recall')
    m.save('../models/iter6_full_userfeat')


if __name__ == '__main__':
    # get data
    df_loan = pd.read_csv(LOAN_PATH)
    df_lender = pd.read_csv(LENDER_PATH)
    sf = gl.SFrame.read_csv(PAIR_PATH, delimiter=',', verbose=False)

    # get features
    loan_feature = get_loan_features(df_loan)
    lender_feature = get_lender_features(df_lender)

    # run model
    run_model(sf, loan_feature, lender_feature)
