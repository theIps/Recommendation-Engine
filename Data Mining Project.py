
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from pyspark import SparkContext, SQLContext
import matplotlib.pyplot as plt
import pyspark.sql.functions as func
plt.rcParams["figure.figsize"] = (20,10)

import pandas as pd
try:
    sc = SparkContext("local", "Simple App")
except ValueError:
    pass


# In[2]:


sql_ctx = SQLContext(sc)


# Reading the data

# In[3]:


train_data = sql_ctx.read.csv('/home/ravibisla/PycharmProjects/DataScience/train_rating.txt', header=True)
train_data.registerTempTable('train_data')
train_data.cache()
test_data = sql_ctx.read.csv('/home/ravibisla/PycharmProjects/DataScience/test_rating.txt', header=True)
test_data.registerTempTable('test_data')
test_data.cache()
# train_review = sql_ctx.read.json('/home/ravibisla/PycharmProjects/DataScience/train_review.json')
# train_review.registerTempTable('train_review')


# In[4]:


train_data.head()


# In[5]:


test_data.head()


# <h1>Data Stats</h1>

# In[6]:


unique_users = train_data.select('user_id').distinct().count()
unique_business = train_data.select('business_id').distinct().count()
matrix_size = unique_users * unique_business
actual_values = train_data.count()
sparcity = 1 - (actual_values/float(matrix_size))
print('Unique Users: %d' % unique_users)
print('Unique Business: %d' % unique_business)
print('Matrix Size: %d' % matrix_size)
print('Actual Size (Train Data Rows): %d' % actual_values)
print('Matrix Sparcity: %.6f' % sparcity)
print('Test Data Rows: %d' % test_data.count())


# <h1>Mean Rating of Training Data</h1>

# In[7]:


train_mean_rating = train_data.agg(func.avg(func.col('rating')))
rating_mean = train_mean_rating.take(1)[0]['avg(rating)']
rating_mean


# <h1>Average - Average Rating of user</h2>

# In[8]:


bias_users_rating = train_data.groupby('user_id').agg(func.avg(func.col('rating')).alias('mean'))
bias_users_rating = bias_users_rating.select('user_id',  'mean', (func.col('mean') - rating_mean).alias('bias'))
bias_users_rating.cache()
bias_users_rating.take(5)


# <h1>Standard Deviation of Rating for all Business</h1>

# In[9]:


bias_business_rating = train_data.groupby('business_id').agg(func.avg(func.col('rating')).alias('mean'))
bias_business_rating = bias_business_rating.select('business_id', 'mean', (func.col('mean') - rating_mean).alias('bias'))
bias_business_rating = bias_business_rating
bias_business_rating.cache()
bias_business_rating.take(5)


# <h1>Comparing Average Rating vs Number of Reviews Relation</h1>

# Finding Average Rating for each user
# 

# In[10]:


user_avg_rating_training = sql_ctx.sql("""SELECT user_id, AVG(rating) AS avg_rating, 
                                            COUNT(user_id) AS no_reviews
                                            FROM train_data GROUP BY user_id""")


# In[11]:


user_avg_df = user_avg_rating_training.toPandas()


# In[12]:


plt.scatter(user_avg_df['no_reviews'], user_avg_df['avg_rating'], s=10, alpha=0.8)
plt.title('User Average Rating vs Number of Reviews')
plt.xlabel('User Number of Reviews')
plt.ylabel('User Average Ratings')
plt.show()


# Finding Average Rating for each Business

# In[13]:


business_avg_rating_training = sql_ctx.sql("""SELECT 
                                            business_id, 
                                            AVG(rating) AS avg_rating, 
                                            COUNT(business_id) AS no_reviews
                                            FROM train_data GROUP BY business_id""")


# In[14]:


business_avg_df = business_avg_rating_training.toPandas()
plt.scatter(business_avg_df['no_reviews'], business_avg_df['avg_rating'], s=10, alpha=0.8)
plt.title('Business Average Rating vs Number of Reviews')
plt.xlabel('Business Number of Reviews')
plt.ylabel('Business Average Ratings')
plt.show()


# <h1>Checking If there are matching rows in training and testing data</h1>

# In[15]:


train_test_join_df = train_data.join(test_data, (train_data.user_id==test_data.user_id) 
                & (train_data.business_id==test_data.business_id))


# In[16]:


train_test_join_df.count()


# No Rows are same

# <h1>Finding Users who always give same ratings with more than no of rating threshold</h1>

# In[17]:


UNIQUE_THRESHOLD = 5


# In[18]:


user_ratings_df = train_data.select('user_id', 'rating')
user_unique_ratings = user_ratings_df.groupby('user_id').agg(func.countDistinct('rating'), func.count('rating'), func.max('rating'))
user_unique_ratings = user_unique_ratings.filter(user_unique_ratings['count(DISTINCT rating)']==1).filter(user_unique_ratings['count(rating)']>=UNIQUE_THRESHOLD)
user_unique_ratings.cache()
print('Total User who always give same ratings: %d' % user_unique_ratings.count())


# In[19]:


test_data.count()


# <h1>Finding Movies which always get same ratings with more than no of rating threshold</h1>

# In[20]:


business_ratings_df = train_data.select('business_id', 'rating')
business_unique_ratings = business_ratings_df.groupby('business_id').agg(func.countDistinct('rating'), func.count('rating'), func.max('rating'))
business_unique_ratings = business_unique_ratings.filter(business_unique_ratings['count(DISTINCT rating)']==1).filter(business_unique_ratings['count(rating)']>=UNIQUE_THRESHOLD)
business_unique_ratings.cache()
print('Total Business which always get same ratings: %d' % business_unique_ratings.count())


# So these business and uses always gives the same rating we can just post thier ratings as well

# <h1>Finding these business and users in test data</h1>

# In[21]:


test_data_filtered_user = test_data.join(user_unique_ratings, (user_unique_ratings['user_id'] == test_data['user_id']))
user_filter = test_data_filtered_user.count()
user_filter


# In[22]:


test_data_filtered_business = test_data.join(business_unique_ratings, (business_unique_ratings['business_id'] == test_data['business_id']))
business_filter = test_data_filtered_business.count()
business_filter


# <h1>Using Spark Mlib ALS (enables scalability)</h1>

# In[23]:


from pyspark.sql import SparkSession, functions, types
spark = SparkSession.builder.appName('example application').getOrCreate()
sc = spark.sparkContext
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
trainDf= train_data

trainDf2 = trainDf.select(trainDf.user_id.cast('int').alias('userid'),trainDf.business_id.cast('int').alias('business_id'),trainDf.rating.cast('float').alias('rating'))
# testDf2 = testDf.select(testDf.user_id.cast('int').alias('userid'),testDf.business_id.cast('int').alias('business_id'))


(training, test) = trainDf2.randomSplit([0.8, 0.2])
# Build the recommendation model using ALS on the training data


# als = ALS(maxIter=20, regParam=0.1,rank=8, userCol="userid", itemCol="business_id", ratingCol="rating",
#           coldStartStrategy="drop")
# als = ALS(maxIter=20, regParam=0.01,rank=20, userCol="userid", itemCol="business_id", ratingCol="rating",
#           coldStartStrategy="drop")  # 1.6653266320794866
als = ALS(maxIter=20, regParam=0.1,rank=20, userCol="userid", itemCol="business_id", ratingCol="rating",
          coldStartStrategy="drop")  # 1.6653266320794866
model = als.fit(training)

# Evaluate the model by computing the RMSE on the test data


predictions = model.transform(test)
#Masking

predpand=predictions.toPandas()
mask = predpand.prediction > 5
mask1 = predpand.prediction < 0
column_name = 'prediction'
predpand.loc[mask, column_name] = 5
predpand.loc[mask1, column_name] = 1
evall= spark.createDataFrame(predpand)

# End Masking
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")

rmse = evaluator.evaluate(evall)
print("Root-mean-square error = " + str(rmse))


# In[24]:


train_data.show(5)


# <h2>TF_IDF Analysis Of Reviews :</h2>

# In[25]:


import math
from textblob import TextBlob as tb

def tf(word, blob):
    return blob.words.count(word) / len(blob.words)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob)

def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)



document1 = tb("""Beautiful space, great location, staff rock. Tiny room, but this was expected. Bathroom amazing. Walls, however, paper thin, which is why I can barely string a sentence together in this review.'),
 Row(id=2, text='First time at this group of hotels. Pretty new, only one in UK, another to open in Edinburgh and one in London. Rooms not very big but great price and location for a weekend in Edinburgh. Rooms clean, comfortable, good shower and free wifi!""")

document2 = tb("""Sometimes the food is spot on and delicious and other times it is quite salty at this location.  Very difficult to get a consistently good meal.  Menu items add up quickly..""")

document3 = tb("""Found this the other night.  It is the PF Chang fast food option and it worked perfectly for us.  Limited menu, but lower prices. Very basic decor, but clean and fast seating.  Lettuce Wraps just as good as Chang's.  Very busy, especially the take out.  Glad to have it close""")

bloblist = [document1, document2, document3]
# bloblist =  [tb(new)]
for i, blob in enumerate(bloblist):
    print("Top words in document {}".format(i + 1))
    scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for word, score in sorted_words[:3]:
        print("Word: {}, TF-IDF: {}".format(word, round(score, 5)))


# <h1>Using Surprise</h1>

# In[26]:


train_df = train_data.toPandas()
test_df = test_data.toPandas()
bias_user_df = bias_users_rating.toPandas()
bias_business_df = bias_business_rating.toPandas()


# In[27]:


bias_user_df = bias_user_df.set_index('user_id', inplace=False)['bias']
bias_business_df = bias_business_df.set_index('business_id', inplace=False)['bias']
bias_business_df.head()


# In[28]:


from surprise import KNNBasic, AlgoBase
from surprise import SVD, Dataset, Reader
from surprise import accuracy, GridSearch
import numpy as np
reader = Reader(rating_scale=(1,5))


# In[29]:


train_dataset = Dataset.load_from_df(train_df[['user_id', 'business_id', 'rating']], reader=reader)


# In[30]:


test_df['rating'] = 0
data_test = Dataset.load_from_df(test_df[['user_id', 'business_id', 'rating']], reader=reader)
test_dataset_prediction = data_test.build_full_trainset()


# In[31]:


test_dataset_prediction = test_dataset_prediction.build_testset()


# <h1>Creating Data Folds</h1>

# In[32]:


train_dataset.split(n_folds=5)


# <h1> Creating an Algo Checker </h1>

# In[33]:


def check_algorithim(algo, data_set):
    for (trainset, testset) in data_set.folds():
        algo.train(trainset)
        predictions = algo.test(testset)
        accuracy.rmse(predictions, verbose=True)
    return algo
        


# <h1>Create An Aglo Tester</h1>

# In[34]:


import time
def test_algorithm(algo, test_set=test_dataset_prediction, test_df=test_df):
    pred = algo.test(test_set, verbose=True)
    all_predictions =  [{
        'user_id': p.uid,
        'business_id': p.iid,
        'rating': p.est
    } for p in pred]
    tm = pd.DataFrame(all_predictions)
    test_merger = test_df[['test_id', 'user_id', 'business_id']]
    new_df = pd.merge(tm, test_merger,  how='inner', left_on=['business_id','user_id'], right_on = ['business_id','user_id'])
    new_df['test_id'] = new_df['test_id'].astype(int)
    new_df = new_df.sort_values(by=['test_id'])
    new_df_result = new_df[['test_id', 'rating']]
    time_str = time.strftime("%Y%m%d-%H%M%S")
    new_df_result.to_csv('%s.csv'%time_str, index=False)
    
    


# <h1>Making a Predictor that just returns mean of all ratings</h1>

# In[35]:


# creating my own class to predict
class Mean_Predictor(AlgoBase):
    
    def __init__(self):
        AlgoBase.__init__(self)
        
    def train(self, trainset):
        AlgoBase.train(self, trainset)
        self.mean = self.trainset.global_mean
        
    def estimate(self, user, business):
        return self.mean   
    
mean_predictor = Mean_Predictor()


# In[36]:


mean_algo = check_algorithim(mean_predictor, train_dataset)


# RMSE: 1.4077
# RMSE: 1.4067
# RMSE: 1.4063
# RMSE: 1.4061
# RMSE: 1.4059

# In[37]:


result_mean = test_algorithm(mean_algo)


# <h1> Making a predictor that returns mean with bias added with ratings in range 1 to 5 </h1>

# In[38]:


# creating my own class to predict
class Mean_Bias_Predictor(AlgoBase):
    
    def __init__(self):
        AlgoBase.__init__(self)
        self.user_bias = dict()
        self.business_bias = dict()
        
    def train(self, trainset):
        AlgoBase.train(self, trainset)
        self.user_bias = bias_user_df
        self.business_bias = bias_business_df
        
    
    def estimate(self, user, business):
        sum_means = self.trainset.global_mean
        div = 1
        user_bias = 0
        business_bias = 0
        if self.trainset.knows_user(user):
            user_id = str(self.trainset.to_raw_uid(user))
            user_bias = self.user_bias[user_id]
            
        if self.trainset.knows_item(business):
            business_id = str(self.trainset.to_raw_iid(business))
            business_bias = self.business_bias[business_id]
        
        return sum_means +  user_bias + business_bias
    
mean_bias_predictor = Mean_Bias_Predictor()


# In[39]:


mean_bias_algo = check_algorithim(mean_bias_predictor, train_dataset)


# RMSE: 1.0947
# RMSE: 1.0947
# RMSE: 1.0963
# RMSE: 1.0946
# RMSE: 1.0957
# 

# In[40]:


result_mean = test_algorithm(mean_bias_algo)


# Using Mean and Bias we get the root mean square of 1.45015

# <h1>Trying Normal SVD first</h1>

# In[41]:


svd_algo = check_algorithim(SVD(n_factors=20, n_epochs=50, lr_all=.005, reg_all=0.05, verbose=False),train_dataset)


# RMSE: 1.3229
# RMSE: 1.3209
# RMSE: 1.3195
# RMSE: 1.3220
# RMSE: 1.3213

# In[41]:


#test_algorithm(svd_algo)


# Increase on Kaggle Score RMSE of 1.31

# Another Try Changing hyperparameters

# In[42]:


svd_algo_2 = check_algorithim(SVD(n_factors=90,n_epochs=30, lr_all=.01, reg_all=0.2, biased=True,verbose=False),train_dataset)


# In[43]:


from surprise import KNNBasic, AlgoBase
from surprise import SVD, Dataset, Reader
from surprise import accuracy, GridSearch
import numpy as np
reader = Reader(rating_scale=(1,5))

def check_algorithim(algo, data_set):
    for (trainset, testset) in data_set.folds():
        algo.train(trainset)
        predictions = algo.test(testset)
        accuracy.rmse(predictions, verbose=True)
    return algo
        
check_algorithim(SVD(n_factors=100, n_epochs=20,init_std_dev =0.04,lr_bu=0.01,lr_bi=0.015,lr_pu=0.01,lr_qi=0.004, reg_bu=0.5,reg_bi=0.08,reg_pu=0.3,reg_qi=0.02,biased=True, verbose=False),train_dataset)


# In[ ]:


RMSE: 1.2669 20
RMSE: 1.2673 20
RMSE: 1.2648 20
RMSE: 1.2669 22
RMSE: 1.2673 22
RMSE: 1.2647 22
RMSE: 1.2671 24
RMSE: 1.2675 24
RMSE: 1.2649 24
RMSE: 1.2672 25
RMSE: 1.2676 25
RMSE: 1.2650 25


RMSE: 1.2672 25
RMSE: 1.2676 25
RMSE: 1.2650 25 
RMSE: 1.2678 28
RMSE: 1.2682 28
RMSE: 1.2656 28
RMSE: 1.2684 30
RMSE: 1.2687 30
RMSE: 1.2661 30
RMSE: 1.2690 32
RMSE: 1.2693 32
RMSE: 1.2667 32


# In[44]:


algo_fin=SVD(n_factors=100, n_epochs=20,init_std_dev =0.04,lr_bu=0.01,lr_bi=0.015,lr_pu=0.01,lr_qi=0.004, reg_bu=0.5,reg_bi=0.08,reg_pu=0.3,reg_qi=0.02,biased=True, verbose=False)
traindata_set = Dataset.load_from_df(train_df[['user_id', 'business_id', 'rating']], reader=reader)
train_set=traindata_set.build_full_trainset()
#check_algorithim(algo_fin, train_dataset)
algo_fin.train(train_set)


# RMSE: 1.2628
# RMSE: 1.2658
# RMSE: 1.2626
# RMSE: 1.2642
# RMSE: 1.2634
# 
# 
# 
# 
# RMSE: 1.2912
# RMSE: 1.2905
# RMSE: 1.2891
# RMSE: 1.2925
# RMSE: 1.2921

# In[99]:


test_algorithm(algo_fin)


# Trying another set of parameters
