import graphlab as gl


PAIR_PATH = '../data/cleaned_lender_loan_pairs.csv'


# run the model
def run_model(sf):
    '''
    INPUT: SFrame pair data
    OUTPUT: None

    Read pair data and run the model, print out result, and save it.
    '''
    # split train test
    train, test = gl.recommender.util.random_split_by_user(sf,
                                                           user_id='lender_id',
                                                           item_id='loan_id',
                                                           item_test_proportion=0.3)

    # compare models
    m = gl.recommender.item_similarity_recommender.create(train,
                                                          user_id='lender_id',
                                                          item_id='loan_id',
                                                          verbose=True)

    print 'Precision Recall========================'
    print m.evaluate_precision_recall(test)
    print '='*10, 'Summary', '='*10
    print m.summary()
    m.save('../models/item_item_cf')


if __name__ == '__main__':
    # load pairs into SFrame
    sf = gl.SFrame.read_csv(PAIR_PATH, header=False, delimiter=',', verbose=False)
    run_model(sf)
