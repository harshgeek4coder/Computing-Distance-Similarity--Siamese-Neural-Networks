import numpy as np
import pandas as pd
import tensorflow as tf

import nltk
import re

#nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow.keras.backend as K




from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM,Bidirectional,Concatenate,BatchNormalization
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.layers import LeakyReLU,Dense,Dropout,Lambda
from tensorflow.keras import metrics

import time

print("Libraries Imported Successfully..")