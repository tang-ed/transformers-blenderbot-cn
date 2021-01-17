from transformers_blenderbot import TFBlenderbotSmallForConditionalGeneration, BlenderbotSmallConfig
import numpy as np
import tensorflow as tf
from tensorflow import keras
from config_ import cfg
from typing import List


config = BlenderbotSmallConfig.from_json_file("config.json")


blen_model = TFBlenderbotSmallForConditionalGeneration(config=config)

npzfile = np.load('cn_train_data.npz')
input_ids = npzfile['arr_0']
input_mask = npzfile['arr_1']
output_ids = npzfile['arr_2']
output_mask = npzfile['arr_3']


inp_shape = input_ids.shape[1]
out_shape = output_ids.shape[1] - 1



def shape_list(tensor: tf.Tensor) -> List[int]:

    dynamic = tf.shape(tensor)

    if tensor.shape == tf.TensorShape(None):
        return dynamic

    static = tensor.shape.as_list()

    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def compute_loss(labels, logits):

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction=tf.keras.losses.Reduction.NONE,
    )
    melted_labels = tf.reshape(labels, (-1,))
    active_loss = tf.not_equal(melted_labels, 0)
    reduced_logits = tf.boolean_mask(tf.reshape(logits, (-1, shape_list(logits)[2])), active_loss)
    labels = tf.boolean_mask(melted_labels, active_loss)
    return loss_fn(labels, reduced_logits)


def models():
    inp = keras.layers.Input(shape=[inp_shape], dtype="int64")
    de_out = keras.layers.Input(shape=[out_shape], dtype="int64")
    inp_mask = keras.layers.Input(shape=[inp_shape], dtype="int32")
    out_mask = keras.layers.Input(shape=[out_shape], dtype="int32")
    out = blen_model(input_ids=inp, decoder_input_ids=de_out, attention_mask=inp_mask, decoder_attention_mask=out_mask, training=True)[0]
    return keras.models.Model(inputs=[inp, de_out, inp_mask, out_mask], outputs=out)


def train():

    train_dataset = tf.data.Dataset.from_tensor_slices(({
                                                            'input_1': input_ids,
                                                            'input_2': output_ids[:, :-1],
                                                            'input_3': input_mask,
                                                            'input_4': output_mask[:, :-1],
                                                        }, output_ids[:, 1:]))
    train_dataset = train_dataset.shuffle(1000).batch(cfg.batch_size)

    total_steps = input_ids.shape[0] // cfg.batch_size * cfg.epoch
    print("总步数：", total_steps)
    cosine_decay = keras.experimental.CosineDecay(
        initial_learning_rate=cfg.lr_rate, decay_steps=total_steps)
    optimizer = keras.optimizers.Adam(cosine_decay)
    model = models()
    model.compile(optimizer=optimizer, loss=compute_loss, metrics=["accuracy"])
    model.summary()

    model.fit(train_dataset, epochs=cfg.epoch)
    model.save("tf_model.h5")


if __name__ == '__main__':
    train()
