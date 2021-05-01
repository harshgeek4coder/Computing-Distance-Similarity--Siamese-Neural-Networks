
from clean_text import process_text

from load_data import load_data

data,data_length=load_data()

def prep_data(data,data_length):

    split_ratio=0.8

    train_size=int(data_length*split_ratio)
    test_size=int(data_length-train_size)


    total_train_corpus=[]
    for i in range(train_size):
        total_train_corpus.append([ process_text(data['question1'][:train_size][i] ), process_text ( data['question2'][:train_size][i] ) ])


    total_test_corpus=[]

    for i in range(train_size,data_length):
        total_test_corpus.append([ process_text ( data['question1'][train_size: ][i] ) , process_text ( data['question2'][train_size:][i] )])
        

        
    train_ques1=[s[0] for s in total_train_corpus]
    train_ques2=[s[1] for s in total_train_corpus]

    test_ques1=[s[0] for s in total_test_corpus]
    test_ques2=[s[1] for s in total_test_corpus]

    train_labels=np.array(data['is_duplicate'][:train_size])
    test_labels=np.array(data['is_duplicate'][train_size:data_length])

    print("Data Prepared Successfully..")

    return train_ques1,train_ques2,test_ques1,test_ques2,train_labels,test_labels





    