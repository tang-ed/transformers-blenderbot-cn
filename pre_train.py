from transformers import TFBlenderbotSmallForConditionalGeneration, BlenderbotSmallTokenizer
import tensorflow as tf
from tensorflow import keras
from config_ import cfg
from typing import List


mname = 'facebook/blenderbot_small-90M'
blen_model = TFBlenderbotSmallForConditionalGeneration.from_pretrained(mname)
tokenizer = BlenderbotSmallTokenizer.from_pretrained(mname)

que = []
ans = []
start_token = "__start__"
end_token = "__end__"


with open(cfg.data_file_v2) as f:
    data = f.read().split("\n")

    state = None
    q = None
    a = None

    for i in range(len(data)):
        if data[i] != "":
            if data[i + 1] != "":
                if q is None:
                    q = data[i]
                    a = data[i+1]
                    que.append(q)
                    ans.append(start_token + " " + a)
                    state = q+" | " + a
                    continue
                if state is not None:
                    q = state
                    a = data[i + 1]
                    state = q + " | " + a
                    que.append(q)
                    ans.append(start_token + " " + a)

            else:
                continue
        else:
            state = None
            q = None
            a = None

inps = tokenizer(que, return_tensors="tf", padding=True, truncation=True)
outps = tokenizer(ans, return_tensors="tf", padding=True, truncation=True)

input_ids = inps["input_ids"]

input_mask = inps["attention_mask"]
output_ids = outps["input_ids"]
output_mask = outps["attention_mask"]

inp_shape = input_ids.shape[1]
out_shape = output_ids.shape[1]
print(input_ids.shape[0])
exit()


class NaturalExpDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """学习率自然数衰减"""
    def __init__(self, initial_learning_rate, decay_steps, decay_rate):
        super().__init__()
        self.initial_learning_rate = tf.cast(initial_learning_rate, dtype=tf.float32)
        self.decay_steps = tf.cast(decay_steps, dtype=tf.float32)
        self.decay_rate = tf.cast(decay_rate, dtype=tf.float32)

    def __call__(self, step):
        return self.initial_learning_rate * tf.math.exp(-self.decay_rate * (step / self.decay_steps))





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
    out = blen_model(input_ids=inp, decoder_input_ids=de_out, attention_mask=inp_mask, decoder_attention_mask=out_mask, training=True, use_cache=False)[0]
    return keras.models.Model(inputs=[inp, de_out, inp_mask, out_mask], outputs=out)


def train():

    train_dataset = tf.data.Dataset.from_tensor_slices(({
                                                            'input_1': input_ids,
                                                            'input_2': output_ids,
                                                            'input_3': input_mask,
                                                            'input_4': output_mask,
                                                        },  output_ids))
    train_dataset = train_dataset.shuffle(1000).batch(cfg.batch_size)

    total_steps = inps["input_ids"].shape[0] // cfg.batch_size * cfg.epoch
    print("总步数：", total_steps)

    natural_exp_decay = NaturalExpDecay(initial_learning_rate=cfg.lr_rate,
                                        decay_steps=total_steps,
                                        decay_rate=1e-7)

    optimizer = keras.optimizers.Adam(natural_exp_decay)
    model = models()
    model.compile(optimizer=optimizer, loss=compute_loss, metrics=["accuracy"])
    model.summary()

    model.fit(train_dataset, epochs=cfg.epoch)
    model.save("tf_model.h5")


if __name__ == '__main__':
    train()
