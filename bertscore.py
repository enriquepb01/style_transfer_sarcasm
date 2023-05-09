from bert_score import score

def calculate_bert_score (input_sentence, output_sentence):
    P, R, F = score([output_sentence], [input_sentence], lang="en")
    return F


print(calculate_bert_score('i love machine learning', 'machine learning is the worst'))