## Preprocessing

import numpy as np
import pandas as pd
import string
import re

def clean_data(file_name, max_samples, min_length, max_length):
    df = pd.read_csv(file_name)
    df = df.sample(frac=1)
    if max_samples != None:
        max_samples = np.min([max_samples, df.shape[0]-1])
        df = df.iloc[0:max_samples]
        
    # Removing unnecessary columns
    df = df.drop(columns=['author', 'subreddit', 'score', 'ups', 'downs', 'date', 'created_utc', 'parent_comment'], axis=1)

    # Removing punctuation and making all data lowercase
    df['comment'] = df['comment'].apply(lambda row: re.sub(r'[^\w\s]', '', str(row).lower()))

    # Removing comments that are too long (>20 words) and too short (<5 words)
    df = df[df['comment'].str.count('\s+').gt(min_length)]
    df = df[df['comment'].str.count('\s+').lt(max_length)]

    

    return df

def train_validate_test_split(df, train_percent=.6, validate_percent=.2, seed=369):
    df = df.sample(frac = 1)
    m = df.shape[0]
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[:train_end,:]
    validate = df.iloc[train_end:validate_end,:]
    test = df.iloc[validate_end:,:]
    return train, validate, test
