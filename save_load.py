from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.models import model_from_json

def save_model(model):

  print("Saving Models ..")
  # Save the trained weights
  model.save_weights('/content/drive/MyDrive/Siamese/Models/100D Adadelta/model_weights.h5')

  # Save the model architecture
  with open('/content/drive/MyDrive/Siamese/Models/100D Adadelta/model_architecture.json', 'w') as f:
    f.write(model.to_json())

  # Save the tokenizer
  with open('/content/drive/MyDrive/Siamese/Models/100D Adadelta/tokenizer.json', 'w') as f:
    f.write(tokenizer.to_json())
       
  print("Models Saved Successfully With Tokenizer!")


from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.models import model_from_json

def load_model():

  print("Loading Models ..")

  with open('/content/drive/MyDrive/Siamese/Models/tokenizer.json') as f:
    tokenizer = tokenizer_from_json(f.read())

    # Model reconstruction from JSON file
  with open('/content/drive/MyDrive/Siamese/Models/model_architecture.json', 'r') as f:
    model = model_from_json(f.read())

     # Load weights into the new model
  model.load_weights('/content/drive/MyDrive/Siamese/Models/model_weights.h5')
    
  print("Loaded Models Successfully !")

  return model, tokenizer