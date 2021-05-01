

def prep_embedding_matrix():

    max_words = 50000     
    word_index = tokenizer.word_index
    glove_dir = '/content/drive/MyDrive/Glove Embeddings/glove.6B.100d.txt'
    embeddings_index = {}


    f = open('/content/drive/MyDrive/Glove Embeddings/glove.6B.100d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))
    print(word_index)

    embedding_dim = 100
    max_words = 50000            
    embedding_matrix = np.zeros((max_words, embedding_dim))

    for word, i in word_index.items():
        if i < max_words:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

    return embedding_matrix