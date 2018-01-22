import pandas as pd
import numpy as np
from datetime import datetime as dt
import time
import graphlab.aggregate as agg
pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import graphlab as gl
from graphlab import SFrame
import matplotlib.pyplot as plt


# Read Training Rating Data
# Total Rows: 2038130
train_rating_df = gl.SFrame.read_csv('train_rating.txt')
# Read Testing Rating Data
# Total Rows: 158024
test_rating_df = gl.SFrame.read_csv('test_rating.txt', sep=",")
test_rating_for_answer_join = test_rating_df[['business_id', 'user_id']]



# Read Sampled Submission
sampled_submission = pd.read_csv('sampled_submission.csv', sep=",")
# Get Total Business and Users in Testing Data
n_users = test_rating_df['user_id'].unique()
n_business = test_rating_df['business_id'].unique()
# Get Total Business and Users in Training Data
test_n_users = train_rating_df['user_id'].unique().shape[0]
test_n_business = train_rating_df['business_id'].unique().shape[0]
max_items_in_utility_matrix = len(n_users)*len(n_business)

print('--Training Data--')
print('Users: %s' % len(n_users))
print('Business:%s' % len(n_business))
print('Matrix Size: %s * %s' % train_rating_df.shape)
print('Expected Utility Matrix Items: %s' % max_items_in_utility_matrix)
print('Matrix Actual Size: %s' % train_rating_df.shape[0])
data_percentage = train_rating_df.shape[0]/float(max_items_in_utility_matrix)
print('Matrix Data Sparsity: %s' % ((1-data_percentage)*100))
print('Total User Pair: %s' % ((len(n_users)*(len(n_users)-1))/2))
print('Total Business Pair: %s' % ((len(n_business)*(len(n_business)-1))/2))
train_test_size = int((train_rating_df.shape[0])*0.1)
print('Training Testing Data Size 10 percent: %s' % str(train_test_size))

# --Training Data--
# Users: 75541
# Business:50017
# Matrix Size: 2038130 * 5
# Expected Utility Matrix Items: 3778334197
# Matrix Actual Size: 2038130
# Matrix Data Sparsity: 99.9460574451
# Total User Pair: 2853183570
# Total Business Pair: 1250825136

def data_pre_processing(df, is_train=False):
    if is_train:
        del df['train_id']
    else:
        del df['test_id']
    # del df['date']
    df['year'] = [int(d[0:4]) for d in df['date']]
    df['date'] = [time.mktime(dt.strptime(d, '%Y-%m-%d').timetuple()) for d in df['date']]

    return df

sf_train = gl.SFrame(data=data_pre_processing(train_rating_df, is_train=True))
final_test = gl.SFrame(data=data_pre_processing(test_rating_df, is_train=False))

# Shuffling the data
x = gl.cross_validation.shuffle(sf_train)
#sf_train, sf_test_sample = sf_train.random_split(.9, seed=5)

def create_user_features(df):
    # Getting User Mean df
    user_rating_mean_df = df[['user_id', 'rating']]
    user_rating_mean_df = user_rating_mean_df.groupby(key_columns='user_id',  operations={
                                    'mean_rating': agg.MEAN('rating'),
                                    'std_rating': agg.STD('rating'),
                                    'distinct_rating': agg.COUNT_DISTINCT('rating'),
                                    'count': agg.COUNT('rating')

                                })
    user_features = gl.SFrame(user_rating_mean_df)
    return user_features


def create_item_features(df):
    # Getting Item mean df
    item_rating_mean_df = df[['business_id', 'rating']]
    item_rating_mean_df = item_rating_mean_df.groupby(key_columns='business_id', operations={
                                    'mean_rating': agg.MEAN('rating'),
                                    'std_rating': agg.STD('rating'),
                                    'distinct_rating': agg.COUNT_DISTINCT('rating'),
                                    'count': agg.COUNT('rating')
                                })
    item_features = gl.SFrame(item_rating_mean_df)
    return item_features


user_info = create_user_features(sf_train)
item_info = create_item_features(sf_train)


#Doing k folds
folds = gl.cross_validation.KFold(sf_train, 5)

for train, valid in folds:
    # Training model
    m = gl.factorization_recommender.create(sf_train, user_id='user_id',
                                            item_id='business_id',
                                            target='rating',
                                            num_factors=20,
                                            max_iterations=20,
                                            sgd_step_size=0.1,
                                            solver='sgd',
                                            user_data=user_info,
                                            item_data=item_info,
                                            regularization=0.2,
                                            linear_regularization=0.01,
                                            side_data_factorization=True,
                                            verbose='false')
    print m.evaluate_rmse(valid, target='rating')

m.save('mode_trained')

loaded_model = gl.load_model('final_30_01_sgd_05.csv')
view = m.predict(final_test)

view.show()


def map_values(prediction):
    if prediction > 5:
        return 5
    elif prediction < 1:
        return 1
    else:
        return prediction



business_with_same_rating = item_info[(item_info['count'] >= 5) & (item_info['distinct_rating'] == 1)]
user_with_same_rating = user_info[(user_info['count'] >= 5) & (user_info['distinct_rating'] == 1)]
result = gl.SFrame.read_csv('test_rating.txt', sep=",")
view = [map_values(v) for v in view]
result = result['test_id', 'rating']
result.save('result.csv')





#

# result.save('abc.csv')

# print(loaded_model.get_default_options())
#m.predict(final_test)
#m.evaluate_rmse(sf_test_sample, target='rating')











# model = gl.load_model('ranking_factorization_recommender_.1')
# predict = model.predict(gl.SFrame(test_rating_df))



# Predicting Score
# prediction = m.evaluate(sf_test)



# train_shuffled = gl.toolkits.cross_validation.shuffle(sf_train, random_seed=1)
#
#

# for train, valid in folds:
#     m = gl.ranking_factorization_recommender.create(sf_train, user_id='user_id',
#                                                     ranking_regularization=0.1,
#                                                     item_id='business_id', target='rating')
#     print m.evaluate(valid)



# With Extra added columns and regularization of 0.12
# Optimization Complete: Maximum number of passes through the data reached.
# Computing final objective value and training RMSE.
#        Final objective value: 0.839013
#        Final training RMSE: 0.475723



# With Out Extra Added Columns and no regularization
# Optimization Complete: Maximum number of passes through the data reached.
# Computing final objective value and training RMSE.
# Final objective value: 1.22228
# Final training RMSE: 0.533115


# With Extra Added Columns
# Optimization Complete: Maximum number of passes through the data reached (hard limit).
# Computing final objective value and training RMSE.
#        Final objective value: 3.85757
#        Final training RMSE: 1.3958




# # Load Training meta data from json
# def load_json_from_file(filename):
#     """
#     Load JSON from a file.
#     @input  filename  Name of the file to be read.
#     @returns Output SFrame
#     """
#     # # Read the entire file into a SFrame with one row
#     sf = gl.SFrame.read_csv(filename, delimiter='', header=False)
#     # sf = sf.stack('X1', new_column_type=[str, int])
#     # # The dictionary can be unpacked to generate the individual columns.
#     # sf = sf.unpack('X1', column_name_prefix='')
#     return sf
# train_meta_sf = load_json_from_file('train_review.json')



# Since only 1% of pairs exist we use triples method for storing utility matrix
# Creating a Utility Matrix using triples method

# triple_index = pd.MultiIndex(levels=[[], []],
#                              labels=[[], []],
#                              names=[u'business_id', u'user_id'])
#
# utility_matrix_normalized = train_rating_df.set_index(['user_id', 'business_id'])
# utility_matrix_normalized = utility_matrix_normalized[['rating']]

# In order to normalize a utility matrix we subtract the average rating of each user
# from the rating converting low rating as negative numbers and high rating as positive numbers


#
#
#
# train_rating_df.apply(add_row_to_utility_matrix, axis=1)





































