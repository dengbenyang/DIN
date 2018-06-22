import pandas as pd
import numpy as np
import random
import pickle

random.seed(1234)

with open('/home/share/wangrc/amazon/remap.pkl', 'rb') as f:
    reviews_df = pickle.load(f)
    cate_list = pickle.load(f)
    user_count, item_count, cate_count, example_count = pickle.load(f)

train_set = []
test_set = []
test_df = reviews_df.head(10)
print(reviews_df.head(10))

for reviewerID, hist in test_df.groupby('reviewerID'):
    # print(reviewerID, hist)
    pos_list = hist['asin'].tolist()
    print('-------------')
    print(pos_list)

    def gen_neg():
        neg = pos_list[0]
        while neg in pos_list:
            neg = random.randint(0, item_count-1)
        return neg
    neg_list = [gen_neg() for i in range(len(pos_list))]

    print(neg_list)
    for i in range(1, len(pos_list)):
        hist = pos_list[:i]
        print('hist :', hist)
        if i != len(pos_list) - 1:
            train_set.append((reviewerID, hist, pos_list[i], 1))
            train_set.append((reviewerID, hist, neg_list[i], 0))
        else:
            label = (pos_list[i], neg_list[i])
            test_set.append((reviewerID, hist, label))
print('train ++++++')
print(train_set)
print('test +++++')
print(test_set)
