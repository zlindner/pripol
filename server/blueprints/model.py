import tensorflow as tf

from flask import Blueprint, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


bp = Blueprint('model', __name__, url_prefix='/model')

# initialize tf for use with gpu
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.InteractiveSession(config=config)

model = load_model('./server/model/cnn.h5')

DATA_PRACTICES = [
    'first_party_collection_use', 'third_party_sharing_collection', 'user_choice_control',
    'international_specific_audiences', 'data_security', 'user_access_edit_deletion',
    'policy_change', 'data_retention', 'do_not_track'
]


# TODO change to GET, somehow pass policy from policy.load()
@bp.route('/predict', methods=['POST'])
def predict():
    policy = request.get_json()['policy']

    if not policy:
        return '', 403

    # policy is now a list of each segment's text
    # convert text -> sequences
    seq = policy_to_sequences(policy)

    # predict the data practice for each policy segment
    y_pred = model.predict(seq)
    y_classes = y_pred.argmax(axis=-1)

    predictions = []

    for i, segment in enumerate(policy):
        predictions.append({'segment': segment, 'data_practice': DATA_PRACTICES[y_classes[i]]})

    return jsonify({'predictions': predictions}), 200    

def policy_to_sequences(policy):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(policy)

    seq = tokenizer.texts_to_sequences(policy)
    seq = pad_sequences(seq, maxlen=100, padding='post')

    return seq
