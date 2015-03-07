# Kipfian

day 5 scrum:

What I did:

1. got rid of duplicates
2. found a tolerable amount of data to run locally, which is 100000, around 18%

Roadblock:

1. not understand why switch the direction of loan-lender could have that improvement

Planning to do:

1. transform the loan-lender data
2. use text feature
3. make plan for the weekend
    1. solve the cold start issue for new user and new loan




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
- [ ] Finish iter n: add text-related features

### Day 5
- [x] Transform data from loan-lenders to lender-loans
- [ ] Retrain iter 3 model after got data loaded
- [ ] Finish iter 4: adding text features

### Day 6
- [ ] Find other features and train

### Day 7
- [ ] Try image processing with [this](http://cs.stanford.edu/people/karpathy/deepimagesent/?hn)
- [ ] Finish online learning part

### Day 8
- [ ] Create a working prototype of the app

### Day 9
- [ ] Create a working prototype of the app

### Day 10
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




