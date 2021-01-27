from transformers import TFBlenderbotSmallModel, BlenderbotSmallConfig, BlenderbotSmallTokenizer
import numpy as np
import tensorflow as tf
from tensorflow import keras

config = BlenderbotSmallConfig.from_json_file("config.json")


blen_model = TFBlenderbotSmallModel(config=config)

npzfile = np.load('train_data.npz')
inputs = npzfile['arr_0']
outputs = npzfile['arr_1']

inp_shape = inputs.shape[1]
out_shape = outputs.shape[1]

def models():
    inp = keras.layers.Input(shape=[None], dtype="int64")
    de_out = keras.layers.Input(shape=[None], dtype="int64")
    inp_mask = keras.layers.Input(shape=[None], dtype="int32")
    out_mask = keras.layers.Input(shape=[None], dtype="int32")
    out = blen_model(input_ids=inp, decoder_input_ids=de_out, attention_mask=inp_mask, decoder_attention_mask=out_mask, training=True)[0]
    logits = keras.layers.Dense(config.vocab_size)(out)

    return keras.models.Model(inputs=[inp, de_out, inp_mask, out_mask], outputs=logits), blen_model

def test_model():
    model, _ = models()
    model.load_weights("models/tf_model.h5")
    tokenizer = BlenderbotSmallTokenizer(vocab_file="vocab.json", merges_file="merges.txt")
    input_data = tokenizer(["你 觉 得 大 学 生 应 该 是 什 么 样 子 的 。"], return_tensors="tf")
    output_ids = tf.expand_dims(1, 0)[None, :]
    output_act = tf.expand_dims(1, 0)[None, :]
    input_ids = input_data["input_ids"]
    input_act = input_data["attention_mask"]

    for i in range(127):
        predictions = model([input_ids, output_ids, input_act, output_act], training=False)

        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if tf.equal(predicted_id[0].numpy(), [2]):
            break

        output_ids = tf.concat([output_ids, predicted_id], axis=-1)

    print("".join(tokenizer.batch_decode(tf.squeeze(output_ids, axis=0)[1:])))








if __name__ == '__main__':
    test_model()
