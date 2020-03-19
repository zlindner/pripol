from flask import Blueprint, request, session
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

bp = Blueprint('model', __name__, url_prefix='/model')

model = load_model('./server/models/cnn.h5')

@bp.route('/predict', methods=['GET'])
def predict():
    print(session['policy'])
    
    # convert list of policy segments to sequences
    #x, vocab = init_sequences(policy)

    # predict the data practice for each policy segment
    #y = model.predict(x)

    #print(y)

    return '', 200

def init_sequences(x):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x)
    vocab = tokenizer.word_index

    x = tokenizer.texts_to_sequences(x)
    x = pad_sequences(x, maxlen=100, padding='post')
    
    return x, vocab
