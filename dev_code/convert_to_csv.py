import pandas as pd
from pymongo import MongoClient
import csv
import os
import json
import re


LENDER_PATH = '../data/lenders/'
LENDER_CSV = '../data/lenders.csv'
LOAN_PATH = '../data/loans/'
LOAN_CSV = '../data/loans.csv'
LENDER_LOAN_CSV = '../data/lenders_loans.csv'


def create_pairs():
    # create & upload lenders_loans pair csv file
    client = MongoClient()
    kiva = client.kiva
    mongo_lenders_loans = kiva.lenders_loans

    # get lenders_loans
    cursor_lenders_loans = mongo_lenders_loans.find({}, {'_id': 0})
    lenders_loans = pd.DataFrame(list(cursor_lenders_loans))
    lenders_loans.dropna(inplace=True)

    # create lender-loan pairs
    with open(LENDER_LOAN_CSV, 'w') as f:
        wr = csv.writer(f, delimiter=',')
        for r in lenders_loans.iterrows():
            for l in r[1]['loan_ids']:
                wr.writerow([r[1]['lender_id'], l])


#clean loans json files, grab what I might need, and create a csv file

def clean_loans():
    # get data from json files
    for f in os.listdir(LOAN_PATH):
        d = json.load(open(LOAN_PATH+f))['loans']
        df = pd.DataFrame.from_dict(d)

        df = df.drop(['basket_amount',
                      'currency_exchange_loss_amount',
                      'delinquent',
                      'payments',
                      'funded_amount',
                      'funded_date',
                      'journal_totals',
                      'name',
                      'tags',
                      'themes',
                      'translator',
                      'video'], axis=1)

        # clean & separate borrowers
        df['gender'] = df['borrowers'].map(
            lambda x: x[0]['gender'] if len(x) == 1 else None)
        df['family'] = df['borrowers'].map(lambda x: 1 if len(x) > 1 else 0)
        df = df.drop(['borrowers'], axis=1)

        # clean & separate description
        df['languages'] = df['description'].map(
            lambda x: 'en' if 'en' in x['languages'] else None)
        df['descriptions'] = df['description'].map(
            lambda x: re.sub(r'\r*\n*', '', x['texts']['en'])
            if 'en' in x['languages'] and x['texts']['en'] else None)
        df = df.drop(['description'], axis=1)

        # clean & separate image
        df['image_id'] = df['image'].map(lambda x: x['id'])
        df['image_template_id'] = df['image'].map(lambda x: x['template_id'])
        df = df.drop(['image'], axis=1)

        # clean location
        df['country'] = df['location'].map(lambda x: x['country'])
        df = df.drop(['location'], axis=1)

        # clean terms

        # grab repayment_term
        df['repayment_term'] = df['terms'].map(lambda x: x['repayment_term'])
        # grab repayment_interval
        df['repayment_interval'] = df['terms'].map(
            lambda x: x['repayment_interval'])
        # earliest scheduled payment date
        df['earliest_scheduled_payment'] = df['terms'].map(
          lambda x: x['scheduled_payments'][0]['due_date'] if x['scheduled_payments'] else None)
        # last scheduled payment date
        df['last_scheduled_payment'] = df['terms'].map(
          lambda x: x['scheduled_payments'][-1]['due_date'] if x['scheduled_payments'] else None)

        # clean use
        df['use'] = df['use'].map(
            lambda x: re.sub(r'\r*\n*', '', x) if x else None)

        df = df.drop(['terms'], axis=1)

        df.to_csv(open(LOAN_CSV, 'a'),
                  encoding='utf-8', index=False, line_terminator='\r\n')


# ##clean lenders json files, grab what I need, and create a csv file

def clean_lenders():
    for f in os.listdir(LENDER_PATH):
        d = json.load(open(LENDER_PATH+f))['lenders']
        dfl = pd.DataFrame.from_dict(d)

        dfl = dfl[['country_code',
                   'image',
                   'invitee_count',
                   'lender_id',
                   'loan_count',
                   'member_since']]

        # clean & separate image
        dfl['image_id'] = dfl['image'].map(lambda x: x['id'])
        dfl['image_template_id'] = dfl['image'].map(lambda x: x['template_id'])
        dfl = dfl.drop(['image'], axis=1)

        dfl.to_csv(open(LENDER_CSV, 'a'),
                   encoding='utf-8', index=False, line_terminator='\r\n')


if __name__ == '__main__':
    create_pairs()
    clean_loans()
    clean_lenders()
