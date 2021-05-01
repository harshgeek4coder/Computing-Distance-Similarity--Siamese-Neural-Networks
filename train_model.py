
def train_model(model,train_pad1,train_pad2,test_pad1,test_pad2):
    epochs=50
    batchsize=128
    start = time.perf_counter()
    print("Training Model..")


    history=model.fit([train_pad1,train_pad2],train_labels,
            validation_data=([test_pad1,test_pad2],test_labels),
            batch_size=batchsize,epochs=epochs,verbose=1)

    print("Model trained successfully..")
    elapsed = time.perf_counter() - start
    print('Elapsed %.3f seconds.' % elapsed)

    return history,model