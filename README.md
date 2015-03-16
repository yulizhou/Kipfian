# Kipfian

- [x] add documentations
- [ ] upload the final model to dropbox
- [ ] add running manual to README

## One-liner
A recommender system for [Kiva.org](http://www.kiva.org). Working prototype: [http://www.kipfian.com](http://www.kipfian.com)

## Motivation & Project Value
The official repayment rate from Kiva is around 98%. But the number is only important to lenders, not loans. If we define a loan's success as getting fully-funded, then the success rate will be 96%. It's still very high but the other 4% is actually over 32721 loans. Even one accounts for the 1% official default rate and assumes all of the 1% are fraud, there're still many loans couldn't get any money. What if they do need helps? What if they didn't finish because of bad timing or people's bias? How can we help loans to maximize the likelihood of success?

There're many ways to achieve the goal. A recommender may help. 

## Data Source
Kiva's data dump contains over 3000 json files covering lenders, loans, and the lending relationship between them, dating back to 2006. It's quite a mess with duplicates and tons of NaNs. 

## Models
The first model I tried is the factorization machine because of two reasons:

1. the data is extremely sparse so that capturing interactions is important.
2. I hope to capture side information like borrower's gender and posted date.

The second model is an item-item collabrative filtering which accounts for interactions. 

## Challenges
**Computational cost**. Factorization Machine is difficult to compute because of the size of the feature matrix. In this project, the feature matrix has 1.27M rows and over 2.4M columns. A simple model with minimum features could run 2-3 hours and the most complex one ran for 5 days. It's not really feasible for a two-week project and Kiva, as a NGO, doesn't necessary have the resources to implement it. 

**Evaluating the model**. Because the target is implicit, RMSE is not feasible. Precision score will be biased because the there're a lot of implicit negative feedbacks. Even if I accounted for those zeros, they don't necssarily mean dislikes. The best metric now is recall, meaning for those loans people like, what's the percentage of them are in the recommendations.

However, the recall may not be the best metric because the goal of a recommender is to increase the click through or conversion. So a better one could be click-through rate or conversion rate. 

## Next Steps
- Run AB tests to examine the effectiveness of models with metrics like CTR
- Address cold start problem better. Currently it recommends the most popular one. A better option could be recommending loans that are almost expired and also almost finished. 
- Get more data and more features, like image processing, NLP for non-English text, and lending transaction details. 
- Have the business goal in mind and consider other ways like a better page composition or marketing campaigns. 



