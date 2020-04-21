import tensorflow as tf
import string
import re
from flask import Blueprint, request, jsonify
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

bp = Blueprint('model', __name__, url_prefix='/model')

# initialize tf for use with gpu (dev only?)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.InteractiveSession(config=config)

model = load_model('./server/model/cnn.h5')

@bp.route('/predict', methods=['POST'])
def predict():
    segments = request.get_json()['segments']

    if not segments:
        return '', 403

    clean = clean_segments(segments)

    # convert text -> sequences
    seq = policy_to_sequences(clean)

    # predict the data practice for each segment
    y_pred = model.predict(seq)
    y_classes = y_pred.argmax(axis=-1)

    predictions = []

    for i, segment in enumerate(segments):
        predictions.append({'segment': segment, 'data_practice': str(y_classes[i])})

    return jsonify({'predictions': predictions}), 200    

def policy_to_sequences(policy):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(policy)

    seq = tokenizer.texts_to_sequences(policy)
    seq = pad_sequences(seq, maxlen=100, padding='post')

    return seq

def clean_segments(segments):
    clean = []
    stop_words = set(stopwords.words('english'))

    for i in range(len(segments)):
        clean.append(segments[i].lower())

        # remove punctuation
        clean[i] = clean[i].translate(str.maketrans('', '', string.punctuation))

        # remove stop words
        clean[i] = ' '.join([word for word in clean[i].split() if word not in stop_words])

        # remove extra whitespace
        clean[i] = re.sub(r' +', ' ', clean[i])
        clean[i] = clean[i].strip()

    return clean
