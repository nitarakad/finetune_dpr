import pandas as pd
import random
from random import shuffle
import math
from tqdm import tqdm

# read csv
df = pd.read_csv('queries_highest_lowest.csv')
queries = df['query']
highest_docs = df['highest_doc']
lowest_docs = df['lowest_doc']
other_docs = []

# loop through and generate other gold standards
for i in tqdm(range(len(queries)), total=len(queries), desc="loop through and generate gold standards"):
    curr_other_docs = []
    # get 6 other gold standard as negative samples
    used_idx = []
    for _ in range(6):
        curr_idx = random.randint(0, len(highest_docs))
        while i != curr_idx and curr_idx not in used_idx:
            curr_idx = random.randint(0, len(queries))
        used_idx.append(curr_idx)
        curr_other_docs.append(highest_docs[curr_idx])
    other_docs.append(curr_other_docs)

# create array with entire dataset
dataset_with_other_docs = []
for q, h, l, o in tqdm(zip(queries, highest_docs, lowest_docs, other_docs), total=len(queries), desc="create new array"):
    dataset_with_other_docs.append((q, h, l, o[0], o[1], o[2], o[3], o[4], o[5]))

# shuffle the data
shuffle(dataset_with_other_docs)

# get train and validation data
#split_at_percent = 0.975
#split_at = math.ceil(split_at_percent * len(dataset_with_other_docs))
train = dataset_with_other_docs#[:split_at]
#val = dataset_with_other_docs[split_at:]

# get dataframes
train_df = pd.DataFrame(train, columns=['query', 'highest_doc', 'lowest_doc', 'o0', 'o1', 'o2', 'o3', 'o4', 'o5'])
#val_df = pd.DataFrame(val, columns=['query', 'highest_doc', 'lowest_doc', 'o0', 'o1', 'o2', 'o3', 'o4', 'o5'])

# create csv
train_df.to_csv('dpr_custom/train.csv')
#val_df.to_csv('dpr_custom/val.csv')

print("created train.csv")








