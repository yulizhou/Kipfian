import pandas as pd
import graphlab as gl
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
import string


PAIR_PATH = '../data/cleaned_lender_loan_pairs.csv'
LOAN_PATH = '../data/cleaned_loans.csv'


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
    vect = TfidfVectorizer(max_features=num_features,
                           stop_words='english', tokenizer=tokenize)
    return vect.fit(descriptions)


def get_loan_features(df):
    '''
    INPUT: data frame of cleaned loan
    OUTPUT: SFrame of loan features

    Get features and convert to SFrame for the model.
    '''
    df = df[['id', 'descriptions']]

    # create tfidf features
    text = df['descriptions'].values
    tfidf = pd.DataFrame(get_vectorizer(text).transform(text).toarray())
    tfidf.columns = tfidf.columns.astype(str)
    tfidf = tfidf.astype(float)
    df = df.drop(['descriptions'], axis=1)

    df = pd.concat([df, tfidf], axis=1, join_axes=[df.index])
    df = df.fillna(0)

    loan_feature = gl.SFrame(df.to_dict(orient='list'))
    loan_feature.rename({'id': 'loan_id'})
    return loan_feature


def run_model(sf, loan_feature):
    '''
    INPUT: SFrame of pair data, SFrame of loan feature data
    OUTPUT: None

    Read pair data and run the model, print out result, and save it.
    '''
    # split train test
    train, test = gl.recommender.util.random_split_by_user(sf,
                                                           user_id='lender_id',
                                                           item_id='loan_id',
                                                           item_test_proportion=0.2)

    # compare models
    m = gl.recommender.ranking_factorization_recommender.create(train,
                                                                user_id='lender_id',
                                                                item_id='loan_id',
                                                                item_data=loan_feature,
                                                                num_factors=20,
                                                                regularization=1e-9,
                                                                binary_target=False,
                                                                max_iterations=20,
                                                                num_sampled_negative_examples=1000,
                                                                verbose=True)

    print m.evaluate(test, metric='precision_recall')
    m.save('../models/iter4_text_feats')


if __name__ == '__main__':
    df_loan = pd.read_csv(LOAN_PATH)
    sf = gl.SFrame.read_csv(PAIR_PATH, delimiter=',', verbose=False)

    loan_feature = get_loan_features(df_loan)

    # run model
    run_model(sf, loan_feature)
