import pandas as pd
import graphlab as gl
from random import random
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize


PAIR_PATH = '../data/cleaned_lender_loan_pairs.csv'
LOAN_PATH = '../data/cleaned_loans.csv'


# ##Feature Engineering
# This iteration will take into account several simple features:
# activity, loan_amount, country, posted_date, sector


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

    raw_features['season'] = \
        raw_features['posted_date'].map(lambda x: convert_to_season(x))
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
    train, test = gl.recommender.util.random_split_by_user(sf,
                                                           user_id='lender_id',
                                                           item_id='loan_id',
                                                           item_test_proportion=0.3)

    # compare models
    models = []
    regs = [0.1, 0.01]
    num_factors = range(4, 10)
    for n in num_factors:
        for r in regs:
            m = gl.recommender.ranking_factorization_recommender.create(train,
                                                                        user_id='lender_id',
                                                                        item_id='loan_id',
                                                                        item_data=loan_feature,
                                                                        num_factors=n,
                                                                        regularization=r,
                                                                        binary_target=True,
                                                                        max_iterations=20,
                                                                        verbose=True)
            models.append(m)

    for i, m in enumerate(models):
        print '='*100
        print 'MODEL ', i
        print m.evaluate(test, metric='precision_recall')
        print '='*10, 'Summary', '='*10
        print m.summary()
        m.save('../models/iter5_'+str(i))


if __name__ == '__main__':
    df_loan = pd.read_csv(LOAN_PATH)
    sf = gl.SFrame.read_csv(PAIR_PATH, delimiter=',', verbose=False)

    loan_feature = get_loan_features(df_loan)

    # run model
    run_model(sf, loan_feature)
