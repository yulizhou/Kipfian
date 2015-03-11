# Kipfian

day 8 scrum:

What I did:


Roadblock:

1. numpy, scipy issues in ubuntu
2. recommender performance is low

Planning to do:

1. finish running tfidf iter 4
    - address numpy & scipy issue in ubuntu
2. try new feautures, iter 5



for the cold start issue:
recommend the newest 10 loans sorted by loan_amount


**Presentation**

- Main idea
- Biz value: understand lenders, help people get loan
- the product (screen shots)
- process
- challenges
    - cleaning
    - graphlab with weird issues
    - lack lender info
    - not updated
- insights (viz), what to show?
    - distribution of features in the chosen model
    - loan amount distribution
    - lender count distribution
    - lender count of loan by gender
    - lender count of loan by country
    - map: loan
    - map: lender
    - map: lender to loan, the lending flow
- Next steps




##To-do
### Day 1
- [x] Store the data / combine files / take a sample
    - [x] Load all data into MongoDB
    - [x] Sample part of the data according to lenders
- [x] EDA, log every question to be answered
    - [x] Univariate
    - [x] Multi-variate
    - [x] How about the number of loans of each lender?
    - [x] How about the number of lenders of each loan?

### Day 2
- [x] Finish iter 1: naive rec sys base model with Dato
- [x] Finish iter 2: use Dato's factorization recommender

### Day 3
- [x] Understand ranking factorization machine
- [x] Finish iter 3: add naive features as side features (date, month, year lended)

### Day 4
- [x] Drop duplicates in mongodb
- [x] Research on how to train the model on new loans and new users in graphlab

### Day 5
- [x] Transform data from loan-lenders to lender-loans
- [x] Retrain iter 3 model after got data loaded

### Day 6
- [x] Design and config workflow with ec2
    - [x] Clean and upload csv
        - [x] loans
        - [x] lenders
        - [x] lenders_loans
    - [x] Do a trial first, how to run on csv files only
    - [x] Touch up all codes into .py and upload
- [x] Learn some factorization machine stuff

### Day 7

### Day 8
- [x] Finish iter 4: adding text features

### Day 9
- [ ] Finish online learning part
- [x] Create a working prototype of the app
- [ ] Start working on visualize for presentation, what to plot? what kind of plot?
- [ ] Finish the front page and structure

### Day 10
- [ ] Visualize for presentation
- [ ] Add visual effect

### Day9
- [ ] 



-------------------




## One-liner

Runnable App: [](#)

## Three-liner


## Motivation & Project Value
Kiva is not like a regular microfinancing servie because the fully-funded loans may possibly have a huge impact on the borrower's life and even more people by providing the field partner with liquity and transfering the risk. I hope to help the borrower side and Kiva by maximizing the likelihood of fully funded. 

## Data Source
Kiva's data snapshot containing 3 directories: `lenders` contains 1615 json files about lender information; `loans` contains 1683 json files about loans; and `loans_lenders` contains 545 json files of lenders for each loan. 

## Challenges in the Project
- The `loans_lenders` data is not formatted suitable for a recommender recommending loans to lenders because it's stored according to loans. It brings serious sparcity. So I transformed it to a lender-loan format. 
- Recommender system is inherently hard to test due to sparcity. I always held out 20% of the data for validation. 
- The famous cold start problem. !!!!!!!!!!!!!! solution?????


## Getting Started

## Basic Model Chosen
Use Graphlab's ranking factorization recommender because we don't have explicit rating to rank. Instead of ratings, the model uses latent factors to rank recommended items according to the likelihood of observing those lender-loan pairs. 

## Metric to Optimize
The metric used to optimize is recall because comparing to the situation where the model recommends stories users won't likely to lend, we really want to raise the exposure of stories which users might be interested so that they're more likely to lend. 

## Iteration 1
item-item similarity


## Iteration 2
factorization without features

## Iteration 3
factorization with features: activity, loan_amount, country, posted_date, sector

## Iteration 4
Tfidf, text features in description

## Things to Improve
- Study user beheviors and engineer features accordingly.
- Run AB test to evaluate the performance. 




