Attemp 0: machine learning models
Best output:
Model: Gradient Boosting
Score: 1.185

Attemp 1: folked from kaggle: emsemble of moving average and median
Score: 0.519


Attemp 2:


Attempt 3： seq2seq model 

seq2seq 
https://www.kaggle.com/c/web-traffic-time-series-forecasting/discussion/43795
learning: 
choice of model: 
1. close to ARIMA model, but more flexible and expressive
2. non-parametric, good for training 
3. flexibility to handle different types of data
4. seq2seq: predict next values, conditioning on joint probability of previous values, 
including past predictions (learn to be conservative to avoid extreme prediction at single step)

feature engineering 
minimized to allow rnn to learn on its own  
1. log1exp for price 
2. quarter to quarter, year to year autocorrelation 


feature preprocessing 
1. normalization for all features, including one hot encoded ones 
// 2. stretch feature to timeseris length - autocorrelations
3. train on random fixed legnth sample 
- eg. 200 day sample in 600 day data, gives choice of 400 days to start with 
- 128 day approx 4 months sample //check for implementation

encoder cuDNN GRU - faster
decoder GRUBlockCell
// attention schema from a year ago average with the weight, 
take important dates from past as features in encoder and decoder
shorter encoder - 60~90 days

lossess and regularization 
?log1p(data)
negative prediction clipped at zero

traing with cocob optimizer 
walk-forward split 

reduce variance 
combine three - ensemble

hyperparameter tuning 
SMAC

Define problem: 

# grocery_forecast
