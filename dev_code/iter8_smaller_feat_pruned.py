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
    '''
    INPUT: array of documents, int of number of features wanted
    OUTPUT: vectorizer

    Create Tf-idf of the input documents.
    '''
    vect = TfidfVectorizer(max_features=num_features, stop_words='english', tokenizer=tokenize)
    return vect.fit(descriptions)


def get_loan_features(df):
    '''
    INPUT: data frame of cleaned loan
    OUTPUT: SFrame of loan features

    Get features and convert to SFrame for the model.
    '''
    df = df[['bonus_credit_eligibility', 'loan_amount', 'lender_count',
             'use', 'gender', 'family', 'country',
             'repayment_interval', 'id', 'posted_year', 'posted_month'
             ]]

    # standardize lender_count
    df['lender_count'] = StandardScaler().fit_transform(df['lender_count'])

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
    '''
    INPUT: data frame of cleaned lender
    OUTPUT: SFrame of lender features

    Get features and convert to SFrame for the model.
    '''
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
    '''
    INPUT: SFrame of pair data, SFrame of loan feature data
    OUTPUT: None

    Read pair data and run the model, print out result, and save it.
    '''
    sf['lender_id'] = sf['lender_id'].astype(str)
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
                                                                max_iterations=3,
                                                                num_sampled_negative_examples=500,
                                                                verbose=True)

    print m.evaluate_precision_recall(test, cutoffs=[100000, 300000, 500000, 700000])
    m.save('../models/iter8_smaller_feat_pruned')


def prune_inactive_lenders(df, threshold=500):
    '''
    INPUT: data frame of lender data, int of the threshhold to prune
    OUTPUT: pruned data frame

    Prune the data.
    '''
    return df[df['loan_count'] > threshold]


def drop_unexsiting_lender_ids(sf, df):
    '''
    INPUT: cleaned SFrame, cleaned data frame
    OUTPUT: a SFrame and a data frame without useless records

    Some lender ids in pair data don't exists in lender data,
    which need to be cleaned.
    '''
    ##### drop unexisting lender id
    lender_ids_in_pairs = list(sf['lender_id'].unique())
    lender_ids_in_lenders = list(df['lender_id'].values)
    lender_ids_intersection = \
        set(lender_ids_in_lenders) & set(lender_ids_in_pairs)

    # drop useless lender_ids in sf
    sf['lender_id'] = \
        sf['lender_id'].apply(lambda x: x if str(x) in lender_ids_intersection else 'None')
    sf = sf[sf['lender_id'] != 'None']
    # drop useless lender_ids in df
    df['lender_id'] = \
        df['lender_id'].map(lambda x: x if x in lender_ids_intersection else None)
    df = df.dropna()
    return sf, df


def drop_unexsiting_loan_ids(sf, df):
    '''
    INPUT: cleaned SFrame, cleaned data frame
    OUTPUT: a SFrame and a data frame without useless records

    Some loan ids in pair data don't exists in loan data,
    which need to be cleaned.
    '''
    ##### drop unexisting loan id
    loan_ids_in_pairs = list(sf['loan_id'].unique())
    df['id'] = df['id'].astype(str)
    loan_ids_in_loans = list(df['id'].values)
    loan_ids_intersection = set(loan_ids_in_loans) & set(loan_ids_in_pairs)

    # drop useless loan_ids in sf
    sf['loan_id'] = \
        sf['loan_id'].apply(lambda x: x if x in loan_ids_intersection else None)
    sf = sf.dropna('loan_id')
    # drop useless loan_ids in df
    df['id'] = df['id'].map(lambda x: x if x in loan_ids_intersection else None)
    df = df.dropna()
    return sf, df


if __name__ == '__main__':
    # get data
    df_loan = pd.read_csv(LOAN_PATH)
    df_lender = pd.read_csv(LENDER_PATH)
    sf = gl.SFrame.read_csv(PAIR_PATH, delimiter=',', verbose=False)

    # pruning
    df_lender = prune_inactive_lenders(df_lender, 5000)
    sf, df_lender = drop_unexsiting_lender_ids(sf, df_lender)
    sf, df_loan = drop_unexsiting_loan_ids(sf, df_loan)

    # get features
    loan_feature = get_loan_features(df_loan)
    lender_feature = get_lender_features(df_lender)

    # run model
    run_model(sf, loan_feature, lender_feature)
