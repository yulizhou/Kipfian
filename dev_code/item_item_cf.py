import graphlab as gl


# ##Feature Engineering
# This iteration will take into account several simple features:
# activity, loan_amount, country, posted_date, sector


# clean pairs data
def clean_pair_date(sf):
    sf.rename({'X1': 'lender_id', 'X2': 'loan_id'})
    sf['loan_id'] = sf['loan_id'].astype(str)
    sf['lender_id'] = sf['lender_id'].astype(str)
    return sf


# run the model
def run_model(sf):
    # split train test
    train, test = gl.recommender.util.random_split_by_user(sf, user_id='lender_id',
                                                           item_id='loan_id',
                                                           item_test_proportion=0.3)

    # compare models
    m = gl.recommender.item_similarity_recommender.create(train,
                                                          user_id='lender_id',
                                                          item_id='loan_id',
                                                          verbose=True)

    print 'Precision Recall========================'
    print m.evaluate_precision_recall(test)
    # print m.evaluate(test, metric='precision_recall')
    print '='*10, 'Summary', '='*10
    print m.summary()
    m.save('item_models/item_item_cf')


if __name__ == '__main__':
    # load pairs into SFrame
    sf = gl.SFrame.read_csv('data/lender_loan_pairs.csv', header=False, delimiter=',', verbose=False)
    sf = clean_pair_date(sf)

    run_model(sf)
