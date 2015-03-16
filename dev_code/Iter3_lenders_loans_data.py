import pandas as pd
import graphlab as gl


PAIR_PATH = '../data/cleaned_lender_loan_pairs.csv'
LOAN_PATH = '../data/cleaned_loans.csv'


# This iteration will take into account several simple features:
# activity, loan_amount, country, posted_date, sector

def get_loan_features(df):
    '''
    INPUT: data frame of cleaned loan
    OUTPUT: SFrame of loan features

    Get features and convert to SFrame for the model.
    '''
    features = df[['id', 'activity', 'sector', 'loan_amount',
                   'country', 'posted_date']]
    features['day_of_year'] = \
        features['posted_date'].map(lambda x: x.timetuple().tm_yday)
    features = features.drop(['posted_date'], axis=1)

    # convert features into SFrame
    loan_feature = gl.SFrame(features.to_dict(orient='list'))
    loan_feature.rename({'id': 'loan_id'})
    return loan_feature


# run the model
def run_model(sf, loan_feature):
    '''
    INPUT: SFrame of pair data, SFrame of loan feature data
    OUTPUT: None

    Read pair data and run the model, print out result.
    '''
    # split train test
    train, test = gl.recommender.util.random_split_by_user(sf,
                                                           user_id='lender_id',
                                                           item_id='loan_id',
                                                           item_test_proportion=0.3)

    # compare models
    models = []
    regs = [0.1, 0.01, 0.001]
    num_factors = range(2, 5)
    for n in num_factors:
        for r in regs:
            m = gl.recommender.ranking_factorization_recommender.create(train,
                                                                        user_id='lender_id',
                                                                        item_id='loan_id',
                                                                        item_data=loan_feature,
                                                                        num_factors=n,
                                                                        regularization=r,
                                                                        binary_target=True,
                                                                        verbose=True)
            models.append(m)

    for i, m in enumerate(models):
        print '='*100
        print 'MODEL ', i
        print m.evaluate(test, metric='precision_recall')


if __name__ == '__main__':
    # load data
    df_loan = pd.read_csv(LOAN_PATH)
    sf = gl.SFrame.read_csv(PAIR_PATH, delimiter=',', verbose=False)

    # get features
    loan_feature = get_loan_features(df_loan)

    # run model
    run_model(sf, loan_feature)
