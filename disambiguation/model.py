__author__ = 'yutao'

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Dropout, Activation, Merge
from keras.optimizers import SGD




# author - paper pairs
X_ap = []
y_ap = []

# author pairs
X_aa = []
y_aa = []

embedding_dim = 100

aff_vocab_size = 0
# embed authors
aff_encoder = Sequential()
aff_encoder.add(Embedding(input_dim=aff_vocab_size,
                            output_dim=100,
                            input_length=100))

title_encoder = Sequential()
title_encoder.add(Embedding(input_dim=100,
                            output_dim=100,
                            input_length=100))

venue_encoder = Sequential()
title_encoder.add(Embedding(input_dim=100,
                            output_dim=100,
                            input_length=100))

pub = Sequential()
pub.add(Merge([venue_encoder, title_encoder], mode='sum'))

author = Sequential()
author.add(Merge([aff_encoder], mode='sum'))

edge = Sequential()
edge.add(Merge([pub, author]))



match = Sequential()
match.add(Merge([aff_encoder, title_encoder],
                mode='dot',
                dot_axes=[(2,), (2,)]))

response = Sequential()
response.add(Merge())