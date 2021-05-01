from libs import *

vocab_size=50000
data,data_length=load_data()

def get_max_seq_len(corpus):
    corpus_len=[]
    
    for question in corpus:
        question_list=(str(question)).split()
        corpus_len.append(len(question_list))

    return max(corpus_len)


max_Seq_len=max(get_max_seq_len(data['question1']),get_max_seq_len(data['question2']))
max_Seq_len=sentence_length

train_ques1,train_ques2,test_ques1,test_ques2,train_labels,test_labels=prep_data(data,data_length)

def get_tokenize_paddings(vocab_size,train_ques1,train_ques2,sentence_length):

    tokenizer=Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(train_ques1+train_ques2)
    word_index=tokenizer.word_index



    train_pad1 = pad_sequences(tokenizer.texts_to_sequences(train_ques1),maxlen=sentence_length)
    train_pad2 = pad_sequences(tokenizer.texts_to_sequences(train_ques2),maxlen=sentence_length)



    test_pad1 = pad_sequences(tokenizer.texts_to_sequences(test_ques1),maxlen=sentence_length)
    test_pad2 = pad_sequences(tokenizer.texts_to_sequences(test_ques2),maxlen=sentence_length)

    print("Word Index and Paddings Imported Successfully..")

    return word_index,train_pad1,train_pad2,test_pad1,test_pad2

