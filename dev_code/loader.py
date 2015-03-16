import json
from os import listdir
from pymongo import MongoClient


client = MongoClient()
kiva = client.kiva

# load loans_lenders data
loans_lenders = kiva.loans_lenders
for file in listdir('data/loans_lenders'):
    d = json.load(open('data/loans_lenders/'+file))['loans_lenders']
    loans_lenders.insert(d)
loans_lenders.ensure_index('id', unique=True, dropDups=True)

# load lenders data
lenders = kiva.lenders
for file in listdir('data/lenders'):
    d = json.load(open('data/lenders/'+file))['lenders']
    lenders.insert(d)
lenders.ensure_index('lender_id', unique=True, dropDups=True)

# load loans data
loans = kiva.loans
for file in listdir('data/loans'):
    d = json.load(open('data/loans/'+file))['loans']
    loans.insert(d)
loans.ensure_index('id', unique=True, dropDups=True)


# load lenders_loans data
lenders_loans = kiva.lenders_loans

# create index first to speed things up
tc = lenders.find({'loan_count': {'$gt': 0}}, {'lender_id': 1, '_id': 0})
lender_ids = list(tc)
lenders_loans.insert(lender_ids)
lenders_loans.ensure_index('lender_id', unique=True, dropDups=True)

# transform and insert
for file in listdir('data/loans_lenders'):
    d = json.load(open('data/loans_lenders/'+file))['loans_lenders']
    for loan_lenders in d:
        if loan_lenders['lender_ids'] is not None:
            for lender in loan_lenders['lender_ids']:
                lenders_loans.update({'lender_id': lender},
                                    {'$addToSet': {'loan_ids': loan_lenders['id']}})
