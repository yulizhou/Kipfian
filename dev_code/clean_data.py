import graphlab as gl
import pandas as pd
from random import random


# clean pairs data
def clean_pair_date(sf):
    sf.rename({'X1': 'lender_id', 'X2': 'loan_id'})
    sf['loan_id'] = sf['loan_id'].astype(str)
    sf['lender_id'] = sf['lender_id'].astype(str)
    return sf


# clean loan data
def clean_loan_data(df):
    # drop duplicate header
    df = df[df['activity'] != 'activity']

    # drop columns
    df = df.drop(['paid_date', 'planned_expiration_date', 'languages'], axis=1)

    # drop duplicates
    df = df.drop_duplicates('id')

    # drop nas
    df = df.dropna(subset=['earliest_scheduled_payment', 'last_scheduled_payment',
                           'repayment_interval', 'posted_date',
                           'status', 'repayment_term', 'use'], how='any')

    # fill paid_amount's na with zero
    df['paid_amount'] = df['paid_amount'].fillna(0)

    # fill genders
    df['gender'].fillna(df[df['gender'].isnull()]['gender']
                        .map(lambda x: 'M' if random() <= 0.39 else 'F'), inplace=True)
    df['gender'] = df['gender'].map(lambda x: 1 if x == 'F' else 0)

    # fill null descriptions with empty string
    df['descriptions'] = df['descriptions'].fillna(0)

    # binaralize bonus credit
    df['bonus_credit_eligibility'] = df['bonus_credit_eligibility'].map(lambda x: 1 if x == 'True' else 0)

    # convert posted_date to year and month
    df['posted_date'] = df['posted_date'].map(
        lambda x: pd.to_datetime(x, format='%Y-%m-%dT%H:%M:%SZ'))
    df['posted_year'] = df['posted_date'].map(lambda x: x.year)
    df['posted_month'] = df['posted_date'].map(lambda x: x.month)
    df.drop(['posted_date'], axis=1, inplace=True)

    # convert some columns to int
    df['lender_count'] = df['lender_count'].astype('int64')
    df['loan_amount'] = df['loan_amount'].astype('int64')
    df['paid_amount'] = df['paid_amount'].astype('float64').astype('int64')
    df['repayment_term'] = df['repayment_term'].astype('float64')

    return df


def clean_lender_data(df):
    df = df[df['lender_id'] != 'lender_id']
    df['invitee_count'] = df['invitee_count'].astype(int)
    df['loan_count'] = df['loan_count'].astype(int)
    return df


def drop_unexsiting_loan_ids(sf, df):
    loan_ids_in_pairs = sorted(list(sf['loan_id'].unique()))
    loan_ids_in_loans = sorted(list(df['id'].values))
    loan_ids_intersection = set(loan_ids_in_loans) & set(loan_ids_in_pairs)
    # drop useless loan_ids in sf
    sf['loan_id'] = sf['loan_id'].apply(lambda x: x if x in loan_ids_intersection else None)
    sf = sf.dropna('loan_id')
    # drop useless loan_ids in df
    df['id'] = df['id'].map(lambda x: x if x in loan_ids_intersection else None)
    df = df.dropna()
    return sf, df


def drop_unexsiting_lender_ids(sf, df):
    lender_ids_in_pairs = sorted(list(sf['lender_id'].unique()))
    lender_ids_in_lenders = sorted(list(df['lender_id'].values))
    lender_ids_intersection = set(lender_ids_in_lenders) & set(lender_ids_in_pairs)
    # drop useless lender_ids in sf
    sf['lender_id'] = sf['lender_id'].apply(lambda x: x if x in lender_ids_intersection else None)
    sf = sf.dropna('lender_id')
    # drop useless lender_ids in df
    df['lender_id'] = df['lender_id'].map(lambda x: x if x in lender_ids_intersection else None)
    df = df.dropna()
    return sf, df


if __name__ == '__main__':
    sf = gl.SFrame.read_csv('data/lender_loan_pairs.csv', header=False, delimiter=',', verbose=False)
    sf = clean_pair_date(sf)

    df_lender = pd.read_csv('data/lenders.csv', delimiter=',')
    df_lender = clean_lender_data(df_lender)
    sf, df_lender = drop_unexsiting_lender_ids(sf, df_lender)
    df_lender.to_csv('data/cleaned_lender.csv', index=False)

    df_loan = pd.read_csv('data/loans.csv', delimiter=',')
    df_loan = clean_loan_data(df_loan)
    sf, df_loan = drop_unexsiting_loan_ids(sf, df_loan)
    df_loan.to_csv('data/cleaned_loans.csv', index=False)

    sf.save('data/cleaned_lender_loan_pairs.csv', format='csv')
