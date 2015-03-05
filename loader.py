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
