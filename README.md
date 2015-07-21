# Kipfian

## One-liner
A recommender system for [Kiva.org](http://www.kiva.org).

## Motivation & Project Value
Lenders on Kiva can enjoy a repayment rate of 98% officially. However, if we take a look from loans' perspective, 4% of them (over 32,000) never got funded, even if one accounting for the 1% official default rate and assuming all of them are fraud. What if they do need help? What if they didn't finish because of bad timing or people's bias? How can we help loans to maximize the likelihood of getting funded?

There're many ways to achieve the goal. I think a recommender delivering proper loans to lenders would definitely help. 

## Data Source
Kiva's data dump contains over 3000 json files covering all lenders, loans, and the lending relationships between them. The data contains tons of duplicates and null values.

## Models
The first model is an item-item collabrative filtering due to the simplicity so that I can have a baseline to compare with.

The second model I tried is the factorization machine due to two reasons:

1. the data is extremely sparse so that capturing interactions is very important.
2. I want to capture side information like gender, posted date, repayment interval, and hidden topics of loan descriptions.

## Challenges
**Computational cost**. Factorization Machine is difficult to compute because of the size of the feature matrix. In this project, the feature matrix has 1.27M rows and over 2.4M columns. A simple model with a small set of features could run for 2-3 hours and more complex ones could run for 5 days. It's not really feasible for a two-week project and for Kiva because it's a NGO with limited budgets. 

**Evaluating models**. Because the target is implicit, RMSE is not feasible. Precision isn't meaningful because no interactions doesn't mean that a lender doesn't like a loan. Recall is better in this situation. It means that among those loans people like, which we denote as 1, what's the percentage of them we can recommend.

## Next Steps
- Run AB tests to examine the effectiveness of models with metrics like CTR
- Address cold start problem better. Currently it recommends the most popular one. A better option could be recommending loans that are almost expired and also almost finished. 
- Get more data like lending transaction details and more features, especially image features.
- Have the business goal in mind and consider other ways like a better page composition or marketing campaigns. 

## How to Run it locally?
1. Clone this repo.
2. Download the model from [my dropbox]() (300+MB), create a `models` directory here, and unzip the model in that directory to be `models/item_item_cf`.
3. Install [Flask](http://flask.pocoo.org/docs/0.10/installation/) if you don't have it.
4. In terminal, go to this repo directory and type `python app.py`. If it shows permission error, type `sudo python app.py` and enter admin password.
5. Open browser and go to `http://0.0.0.0:6969/`.
