## Attemp 0: machine learning models
Best output:
Model: Gradient Boosting
Score: 1.185

## Attemp 1: folked from kaggle: emsemble of moving average and median
Score: 0.519

## Attemp 2: simple lstm model
### Data: two dataframe: 
- time-series data only, each store+item combination takes one line, each day takes one column (train_pivot.feat), log1exp for all the unit_sales
- item and store features merged from stores.csv and items.csv, one hot encoded, all normalized

### Process data for LSTM
- Timesteps: 5
- Predict span: 10 
- Validation set: 115 days (because of the the timesteps, there will be 5 days shifting,  setting number of days to be encoded be 100 and number of days to predict be 10)
- Training set: 292 days (each batch will take 100 days)

Experiment logs
Model 0:
- LSTM: 126 units
- LSTM: 64 units
- Merge features input 
- Dense: 64 units relu activation 
- Dense: 10 units relu activation

Results: nan loss (still not fixed), tried following 
- normalize with z-score
- min max normalize
- change to GRU (suspect gradient exploding)
- change to a smaller batch
- change to a smaller network (with dropouts)
- use only one timeseries, still nan loss, while accuracy is 1 
 

### Attempt 3： seq2seq model (did not start)

seq2seq case study from one Kaggle competition winner team
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
1. log1exp for sales 
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

