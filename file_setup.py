from preprocessing import clean_data, train_validate_test_split
import pandas as pd

def write_data_file(df, filename):
    with open(filename, 'w') as f:
        for idx in df.index:
            f.write(df['comment'][idx] + '\n')

df = clean_data('C:\\Users\\enriq\\git\\style_transfer_sarcasm\\train-balanced-sarcasm.csv', 
                              max_samples=1000000, 
                              min_length=2, 
                              max_length=21)

df_0 = df.loc[df.label == 0]
df_1 = df.loc[df.label == 1]

train_0, validate_0, test_0 = train_validate_test_split(df_0)
train_1, validate_1, test_1 = train_validate_test_split(df_1)

write_data_file(train_0, 'sarcasm_data/train.0')
write_data_file(validate_0, 'sarcasm_data/dev.0')
write_data_file(test_0, 'sarcasm_data/test.0')
write_data_file(train_1, 'sarcasm_data/train.1')
write_data_file(validate_1, 'sarcasm_data/dev.1')
write_data_file(test_1, 'sarcasm_data/test.1')
