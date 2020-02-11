from tensorflow.keras.models import model_from_json




def model(model_path):
    model_path_save = model_path
    #base_model = InceptionV3(weights='imagenet', include_top=False)
    #model = load_model(model_path_save)
    #return model
    
    #base_model = InceptionV3(weights='imagenet', include_top=False)
    #model = load_model(model_path_save)

    # load json and create model
    json_file = open(model_path_save, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    #model.load_weights(model_path_save)
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX Loaded model from disk")
    return model

model('models/transfer_inceptionV3.json')