from test_inference import create_test_data
from save_load import load_model

model,tokenizer=load_model()

test_data_1="ENTER FIRST SENTENCE HERE"
test_data_2="ENTER SECOND SENTENCE HERE"

max_Seq_len=237
test_data_1, test_data_2=create_test_data(tokenizer=tokenizer,test_sentences_pair=test_sentence_pairs,max_sequence_length=max_Seq_len)

get_predictions(test_data_1, test_data_2,model)