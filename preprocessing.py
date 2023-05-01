## Preprocessing

import numpy as np
import pandas as pd
import string
import re

def clean_data(file_name)
    df = pd.read_csv(file_name)
    # Removing unnecessary columns
    df = df.drop(columns=['author', 'subreddit', 'score', 'ups', 'downs', 'date', 'created_utc', 'parent_comment'], axis=1)

    # Removing punctuation and making all data lowercase
    df['comment'] = df['comment'].apply(lambda row: re.sub(r'[^\w\s]', '', str(row).lower()))

    # Removing comments that are too long (>20 words) and too short (<5 words)
    df = df[df['comment'].str.count('\s+').gt(4)]
    df = df[df['comment'].str.count('\s+').lt(21)]

    df.to_csv('clean_data.csv')