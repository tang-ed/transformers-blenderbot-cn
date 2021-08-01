from BlenderbotSmall import TFBlenderbotSmallForConditionalGeneration, BlenderbotSmallConfig
import numpy as np
import tensorflow as tf
from tokenizer import SelfTokenizer


tokenizer = SelfTokenizer("vocab.json")


config = BlenderbotSmallConfig.from_json_file("model_file/config_small.json")


b_model = TFBlenderbotSmallForConditionalGeneration(config=config)

inp = tokenizer.encoder(["你好啊，我叫唐小书。"], return_tensor="tf")
lab = tokenizer.encoder(["唐小书，你好，我叫唐恩达，是来自中国，你也哪里人呢？"], return_tensor="tf")
model_inp = {
    "input_ids":inp["input_ids"],
    "attention_mask":inp["attention_mask"],
    "decoder_input_ids":lab["input_ids"],
    "decoder_attention_mask":lab["attention_mask"]
}

b_model(model_inp)


b_model.summary()


class Blenderbot(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.b_model = b_model

    def get_b_model(self):
        return self.b_model

    def call(self, input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        training=None, **kwargs):

        lm_logits = self.b_model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, training=training).logits

        return lm_logits


model = Blenderbot()

model(**model_inp)

model.load_weights("model.h5")

blenderbot = model.get_b_model()

blenderbot.save_pretrained("blenderbot-model")
