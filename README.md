# style_transfer_sarcasm
EE 499 final project. Enrique Perez &amp; Justin Sado

Data used -> https://paperswithcode.com/dataset/sarc (train-balanced-sarcasm.csv)

preprocessing.py -> helper functions to clean up our data

mlp_classifier.py and cnn_classifier.py -> End-to-end implementation of classifiers

file_setup.py -> Script to put data in appropriate format for GAN

bertscore.py -> Implementing BERTScore for GAN evaluation

pickle_to_graph.py -> Script to display accuracy and loss of a model over epochs

The following code was referenced in writing cnn_classifier.py & mlp_classifier.py:
	https://github.com/keithchugg/ee499_ml_spring23/blob/main/nnet_notebooks/03_fmnist_torch.py#fromHistory
	https://github.com/Nielspace/BERT/blob/master/BERT%20Text%20Classification%20fine-tuning.ipynb
