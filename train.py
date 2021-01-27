from transformers import TFBlenderbotSmallModel, BlenderbotSmallConfig
import numpy as np
import tensorflow as tf
from tensorflow import keras
from config_ import cfg


config = BlenderbotSmallConfig.from_json_file("config.json")


blen_model = TFBlenderbotSmallModel(config=config)

npzfile = np.load('train_data.npz')
inputs = npzfile['arr_0']
outputs = npzfile['arr_1']

inp_shape = inputs.shape[1]
out_shape = outputs.shape[1]



class NaturalExpDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """学习率自然数衰减"""
    def __init__(self, initial_learning_rate, decay_steps, decay_rate):
        super().__init__()
        self.initial_learning_rate = tf.cast(initial_learning_rate, dtype=tf.float32)
        self.decay_steps = tf.cast(decay_steps, dtype=tf.float32)
        self.decay_rate = tf.cast(decay_rate, dtype=tf.float32)

    def __call__(self, step):
        return self.initial_learning_rate * tf.math.exp(-self.decay_rate * (step / self.decay_steps))

def models():
    inp = keras.layers.Input(shape=[inp_shape], dtype="int64")
    de_out = keras.layers.Input(shape=[out_shape-1], dtype="int64")
    inp_mask = keras.layers.Input(shape=[inp_shape], dtype="int32")
    out_mask = keras.layers.Input(shape=[out_shape-1], dtype="int32")
    out = blen_model(input_ids=inp, decoder_input_ids=de_out, attention_mask=inp_mask, decoder_attention_mask=out_mask, training=True)[0]
    logits = keras.layers.Dense(config.vocab_size)(out)

    return keras.models.Model(inputs=[inp, de_out, inp_mask, out_mask], outputs=logits), blen_model


def train():

    dataset = ({
            'input_1': inputs,
            'input_2': outputs[:, :-1],
            'input_3': np.array(inputs > 0, dtype="int32"),
            'input_4': np.array(outputs[:, :-1] > 0, dtype="int32"),
            }, outputs[:, 1:])

    train_dataset = tf.data.Dataset.from_tensor_slices(dataset)
    train_dataset = train_dataset.shuffle(2000).batch(cfg.batch_size)

    total_steps = inputs.shape[0] // cfg.batch_size * cfg.epoch
    print("总步数：", total_steps)
    natural_exp_decay = NaturalExpDecay(initial_learning_rate=cfg.lr_rate,
                                        decay_steps=total_steps,
                                        decay_rate=1e-6)

    optimizer = keras.optimizers.Adam(natural_exp_decay)
    model, b_model = models()
    model.compile(optimizer=optimizer, loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE), metrics=["accuracy"])
    model.summary()


    model.fit(train_dataset, epochs=cfg.epoch)
    model.save_weights("tf_model.h5")
    blen_model.save_pretrained("blen_model")


if __name__ == '__main__':
    train()
