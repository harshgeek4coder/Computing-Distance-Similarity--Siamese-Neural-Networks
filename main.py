from libs import *

from load_data import load_data
from data_prep import prep_data
from tokenize_padding import get_tokenize_paddings
from prep_embed import prep_embedding_matrix
from model import siamese_manhattan_network
from train_model import train_model
from save_load import save_model,load_model


data,data_length=load_data()

train_ques1,train_ques2,test_ques1,test_ques2,train_labels,test_labels=prep_data(data,data_length)

word_index,train_pad1,train_pad2,test_pad1,test_pad2=get_tokenize_paddings(vocab_size,train_ques1,train_ques2,sentence_length)

embedding_matrix=prep_embedding_matrix()

model=siamese_manhattan_network(embedding_matrix)

history,model=train_model(model,train_pad1,train_pad2,test_pad1,test_pad2)

save_model(model)








