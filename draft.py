

df_sub = df.sample(n=1000, replace=True)

       
import time

start = time.time()

id_u = list(sorted(df_sub.id.unique()))
date_u = list(sorted(df_sub.date_str.unique()))
row = pd.Categorical(df_sub.id).codes 
col = pd.Categorical(df_sub.date_str).codes

data = df_sub['unit_sales_cal'].tolist()
sparse_matrix = csr_matrix((data, (row, col)), shape=(len(id_u), len(date_u)))
df_sub = pd.SparseDataFrame([ pd.SparseSeries(sparse_matrix[i].toarray().ravel(), fill_value=0) 
	for i in np.arange(sparse_matrix.shape[0])], index=id_u, columns=date_u, default_fill_value=0)

end = time.time()
print(end - start)




CREATE table train_sales_temp (
	ind INT NOT NULL AUTO_INCREMENT
	, id INT
	, date VARCHAR(255)
	, unit_sales_cal double
	, primary key(ind));

drop table train_sales_sub;


SET @sql = NULL;

SELECT
  GROUP_CONCAT(DISTINCT
    CONCAT(
      "SUM(IF(date = '",
      date,
      "', unit_sales_cal, 0)) AS '",
      date, "'"
    )) INTO @sql
FROM train_sales_sub2;

SET @sql = CONCAT('create view pivot_sub_view as SELECT id, ', @sql, ' FROM train_sales_sub2 GROUP BY id');

PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;


select * FROM train_sales_sub2 where date = '2017-07-20' limit 10;


PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;


SET @sql0 = NULL;

SELECT
  GROUP_CONCAT(DISTINCT
    CONCAT(
      "MAX(IF(delta=",
      delta,
      ",unit_sales_cal,0)) AS '",
      date, "'"
    )) INTO @sql0
FROM train_sales_sub2;

select @sql;

SET @sql = CONCAT('create view pivot_sub_view as SELECT id, ', @sql, ' FROM train_sales_sub2 GROUP BY id');
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

df['min'] = '2016-07-01'
# train_sub.feat -> pivoted from 100000 datapoints 


columns = ['id', 'date', 'unit_sales_cal']
df = df[columns]

df['date']=df['date'].dt.strftime('%Y-%m-%d')

print ('pivoting table..')
# sparse matrix, need to optimize performance and avoid "negative dimensions" error
from scipy.sparse import csr_matrix
id_u = list(sorted(df.id.unique()))
date_u = list(sorted(df.date.unique()))
row = pd.Categorical(df.id).codes 
col = pd.Categorical(df.date).codes

data = df['unit_sales_cal'].tolist()
sparse_matrix = csr_matrix((data, (row, col)), shape=(len(id_u), len(date_u)))
df = pd.SparseDataFrame([ pd.SparseSeries(sparse_matrix[i].toarray().ravel(), fill_value=0) 
                      for i in np.arange(sparse_matrix.shape[0])], 
               index=id_u, columns=date_u, default_fill_value=0)




########## sample data 
# originally: 41839709 41852644
df['uid'] = df_ori['store_nbr'].astype(str)+'_' + df_ori['item_nbr'].astype(str)

min_date = min(df.date)
df['min'] = min_date
df['delta'] = (df.date - df['min']).dt.days

columns = ['uid', 'delta', 'unit_sales_cal']
df = df[columns]

# 171101 uid 
import random 
uid_list = np.unique(df.uid)
random.shuffle(uid_list)
uid_rand = random.choices(uid_list, k=1000)
# rand_nums = random.sample(range(len(uid_list), 1000)
# uid_rand = uid_list[rand]

# number of entries: 246071
df_sub = df[df.uid.isin(uid_rand)]
df_sub.to_feather('data/train_days_sub.feat')


##### pivot table 
from pyspark import SparkContext
from pyspark import SQLContext
from pyspark.sql.functions import first


df_sub = pd.read_csv('train_days_sub.csv')

sc = SparkContext()
sqlContext = SQLContext(sc)
spark_df = sqlContext.createDataFrame(df_sub)

pivot_df = spark_df.groupBy('uid').pivot('delta').agg(first('unit_sales_cal'))

pivot_df = pivot_df.toPandas()

names = pd.date_range(start=min_date, periods=411,  freq='D') 
c_names = names.strftime('%Y-%m-%d')
c_names = ['uid'] + [str(i) for i in c_names]
# missing 1 day: day 177
c_names = c_names[:176]+c_names[177:]

pivot_df.columns = c_names

pivot_df.to_feather('data/train_pivot.feat')
