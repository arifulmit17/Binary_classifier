First open the file Binary_classifier.py


import preprocess.py file

give file path to pd.read_csv('training data file path')
for example,
train_data=pd.read_csv('training_data_original.csv')



To train the classifier without any pretrained word embedding,
import binary_classifier_rnn_word2vec.py file


or to train the classifier with pretrained word embeddings,
import binary_classifier_rnn.py 

save and close the file.

if using pretrained word embeddings,
Open the file binary_classifier_rnn.py 

to use pretrained word2vec embeddings put file path in line,
for example,

word_vectors = KeyedVectors.load_word2vec_format('C:\\Users\\User\\Documents\\resources\\Binary_classifier\\GoogleNews-vectors-negative300.bin', binary=True)

save and close the file

the GoogleNews-vectors-negative300.bin can be downloaded from,

http://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/
