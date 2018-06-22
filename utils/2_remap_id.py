import random
import pickle
import numpy as np

random.seed(1234)

with open('/home/share/wangrc/amazon/reviews.pkl', 'rb') as f:
    reviews_df = pickle.load(f)
    reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]
with open('/home/share/wangrc/amazon/meta.pkl', 'rb') as f:
    meta_df = pickle.load(f)
    meta_df = meta_df[['asin', 'categories']]
    meta_df['categories'] = meta_df['categories'].map(lambda x: x[-1][-1])


def build_map(df, col_name):
    key = sorted(df[col_name].unique().tolist())
    m = dict(zip(key, range(len(key))))
    df[col_name] = df[col_name].map(lambda x: m[x])
    return m, key


asin_map, asin_key = build_map(meta_df, 'asin')
cate_map, cate_key = build_map(meta_df, 'categories')
revi_map, revi_key = build_map(reviews_df, 'reviewerID')
print(asin_map[asin_key[0]])

user_count, item_count, cate_count, example_count =\
    len(revi_map), len(asin_map), len(cate_map), reviews_df.shape[0]
print('user_count: %d\titem_count: %d\tcate_count: %d\texample_count: %d' %
      (user_count, item_count, cate_count, example_count))

print(meta_df.head(10))
meta_df = meta_df.sort_values('asin')
meta_df = meta_df.reset_index(drop=True)

print(meta_df.head(10))
print('-----------reviews-----------------')
print(reviews_df.head(10))
reviews_df['asin'] = reviews_df['asin'].map(lambda x: asin_map[x])
print(reviews_df.head(10))
reviews_df = reviews_df.sort_values(['reviewerID', 'unixReviewTime'])
print(reviews_df.head(10))
reviews_df = reviews_df.reset_index(drop=True)
print(reviews_df.head(10))
reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]
print('-------final reviews------------')
print(reviews_df.head(10))

cate_list = [meta_df['categories'][i] for i in range(len(asin_map))]
cate_list = np.array(cate_list, dtype=np.int32)


with open('/home/share/wangrc/amazon/remap.pkl', 'wb') as f:
    pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL)  # uid, iid
    pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)  # cid of iid line
    pickle.dump((user_count, item_count, cate_count, example_count),
                f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((asin_key, cate_key, revi_key), f, pickle.HIGHEST_PROTOCOL)
