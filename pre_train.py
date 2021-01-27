from transformers import TFBlenderbotSmallModel, BlenderbotSmallTokenizer, BlenderbotSmallConfig
import tensorflow as tf
from tensorflow import keras
from config_ import cfg
import numpy as np


mname = 'facebook/blenderbot_small-90M'
blen_model = TFBlenderbotSmallModel.from_pretrained(mname)
tokenizer = BlenderbotSmallTokenizer.from_pretrained(mname)
config = BlenderbotSmallConfig.from_pretrained(mname)


que = []
ans = []
start_token = 1
end_token = 2


with open(cfg.data_file) as f:
    data = f.read().split("\n")

    state = None
    q = None
    a = None

    for i in range(len(data)):
        if data[i] != " " and data[i] != "":
            if data[i + 1] != "" and data[i+1] != " ":
                if q is None:
                    q = data[i]
                    a = data[i+1]
                    que.append(q)
                    ans.append(a)

                    state = q+" | " + a
                    continue
                if state is not None:
                    q = state
                    a = data[i + 1]
                    state = q + " | " + a
                    que.append(q)
                    ans.append(a)


            else:
                continue
        else:
            state = None
            q = None
            a = None

inps = tokenizer(que, return_tensors="tf", padding=True, truncation=True)
outputs = tokenizer(ans, return_tensors="tf", padding=True, truncation=True, max_length=126)

ans_datas = outputs["input_ids"]

new_ans = []
for i in ans_datas:
    ans_ls = list(i.numpy())
    ans_ls.insert(0, start_token)
    try:
        z_index = ans_ls.index(0)
        ans_ls.insert(z_index, end_token)
    except:
        ans_ls.insert(-1, end_token)
    new_ans.append(ans_ls)

ans_data = np.array(new_ans)

input_ids = inps["input_ids"]

input_mask = inps["attention_mask"]
output_ids = ans_data
output_mask = np.array(output_ids > 0, dtype="int32")

inp_shape = input_ids.shape[1]
out_shape = output_ids.shape[1]



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
            'input_1': input_ids,
            'input_2': output_ids[:, :-1],
            'input_3': input_mask,
            'input_4': output_mask[:, :-1],
            }, output_ids[:, 1:])

    train_dataset = tf.data.Dataset.from_tensor_slices(dataset)
    train_dataset = train_dataset.shuffle(2000).batch(cfg.batch_size)

    total_steps = input_ids.shape[0] // cfg.batch_size * cfg.epoch
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
