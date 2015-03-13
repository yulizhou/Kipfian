# Kipfian

**Presentation**

Improvement:

- Shorten the first part
- Shorten the Next Step part
- Add comparison
    - Which features are important? How to decide?
    - precision / recall / which one to optimize / AUC
- Custom domain name


compare to benchmark of randomly choosing


Adam's advice:
doesn't have to use FM
show process of choosing between models
show knowledge about the tradeoff
show features used
show good next steps




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
- [x] Create a working prototype of the app
- [x] Start working on visualize for presentation, what to plot? what kind of plot?
- [x] Finish the front page and structure
- [x] Run simple item-item CF to compare

### Day 10
- [x] test app locally
- [x] Add user features to the model

### Day 11
- [x] the diff b/w SVD and factorization in machine
- [x] Among most lended loans, what's the topics in story? Word cloud?
- [x] Write profile for Zack
- [x] Finish ppt

### Day 12
- [ ] Run models with diff set of features
- [ ] Address cold start problem
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
The `loans_lenders` data is not formatted suitable for a recommender recommending loans to lenders because it's stored according to loans. It brings serious sparcity. So I transformed it to a lender-loan format. 

Recommender system is inherently hard to test due to sparcity. I always held out 20% of the data for validation. 


## Getting Started

## Basic Model Chosen
Use Graphlab's ranking factorization recommender because we don't have explicit rating to rank. Instead of ratings, the model uses latent factors to rank recommended items according to the likelihood of observing those lender-loan pairs. 

## Metric to Optimize
The metric used to optimize is recall because comparing to the situation where the model recommends stories users won't likely to lend, we really want to raise the exposure of stories which users might be interested so that they're more likely to lend. 

## Things to Improve
- Study user beheviors and engineer features accordingly.
- Run AB test to evaluate the performance. 




