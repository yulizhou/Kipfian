import json
from os import listdir
from pymongo import MongoClient
from random import random


client = MongoClient()
kiva = client.kiva

# load loans_lenders data
loans_lenders = kiva.loans_lenders
for file in listdir('data/loans_lenders'):
    d = json.load(open('data/loans_lenders/'+file))['loans_lenders']
    for data in d:
        data['rnd'] = random()
    loans_lenders.insert(d)
    loans_lenders.ensure_index({'rnd': 1})

# load lenders data
lenders = kiva.lenders
for file in listdir('data/lenders'):
    d = json.load(open('data/lenders/'+file))['lenders']
    for data in d:
        data['rnd'] = random()
    lenders.insert(d)
    lenders.ensure_index('rnd')

# load loans data
loans = kiva.loans
for file in listdir('data/loans'):
    d = json.load(open('data/loans/'+file))['loans']
    for data in d:
        data['rnd'] = random()
    loans.insert(d)
    loans.ensure_index('rnd')
