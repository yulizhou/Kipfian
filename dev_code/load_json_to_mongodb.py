import json
from os import listdir
from pymongo import MongoClient


LOAN_LENDER_PATH = '../data/loans_lenders/'
LENDER_PATH = '../data/lenders/'
LOAN_PATH = '../data/loans/'


def load_loans_lenders(loans_lenders):
    for file in listdir(LOAN_LENDER_PATH):
        d = json.load(open(LOAN_LENDER_PATH+file))['loans_lenders']
        loans_lenders.insert(d)
    loans_lenders.ensure_index('id', unique=True, dropDups=True)


def load_lenders(lender):
    for file in listdir(LENDER_PATH):
        d = json.load(open(LENDER_PATH+file))['lenders']
        lenders.insert(d)
    lenders.ensure_index('lender_id', unique=True, dropDups=True)


def load_loans(loans):
    for file in listdir(LOAN_PATH):
        d = json.load(open(LOAN_PATH+file))['loans']
        loans.insert(d)
    loans.ensure_index('id', unique=True, dropDups=True)


def transform_lenders_loans(lenders_loans):
    # create index first to speed things up
    tc = lenders.find({'loan_count': {'$gt': 0}}, {'lender_id': 1, '_id': 0})
    lender_ids = list(tc)
    lenders_loans.insert(lender_ids)
    lenders_loans.ensure_index('lender_id', unique=True, dropDups=True)

    # transform and insert
    for file in listdir(LOAN_LENDER_PATH):
        d = json.load(open(LOAN_LENDER_PATH+file))['loans_lenders']
        for loan_lenders in d:
            if loan_lenders['lender_ids'] is not None:
                for lender in loan_lenders['lender_ids']:
                    lenders_loans.update({'lender_id': lender},
                                        {'$addToSet': {'loan_ids': loan_lenders['id']}})


if __name__ == '__main__':
    # create MongoDB connections
    client = MongoClient()
    kiva = client.kiva

    # load loans_lenders data
    loans_lenders = kiva.loans_lenders
    load_loans_lenders(loans_lenders)

    # transform to lenders_loans data
    lenders_loans = kiva.lenders_loans
    transform_lenders_loans(lenders_loans)

    # load lenders data
    lenders = kiva.lenders
    load_lenders(lenders)

    # load loans data
    loans = kiva.loans
    load_loans(loans)
