from tokenizer import SelfTokenizer
from BlenderbotSmall import TFBlenderbotSmallForConditionalGeneration, BlenderbotSmallConfig
import tensorflow as tf

tokenizer = SelfTokenizer(vocab_file="vocab.json")
model = TFBlenderbotSmallForConditionalGeneration.from_pretrained("blenderbot-model")
encoder = model.get_encoder()

def test_model(sentence):


    input_data = tokenizer.encoder([sentence], add_special_tokens=True, return_tensor="tf")

    input_ids = input_data["input_ids"]
    att = input_data["attention_mask"]

    encoder_out = encoder(input_ids=input_ids, attention_mask=att)
    decoder_inp = tf.ones((1, 1), dtype="int64")

    for i in range(50):
        predictions = model(input_ids = input_ids, decoder_input_ids = decoder_inp, encoder_outputs=encoder_out, attention_mask=att)[0]

        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), "int64")

        if tf.equal(predicted_id[0], [2]):
            break

        decoder_inp = tf.concat([decoder_inp, predicted_id], axis=-1)
    decoder_inp = tf.squeeze(decoder_inp, axis=0)
    result = "".join(tokenizer.decoder(decoder_inp)[0])
    return result


def main(out="", long_dialogue=False, dialogue_num=2):
    sentence_ls = []
    while True:
        sentence = str(input("唐恩达："))
        if long_dialogue:
            if len(sentence_ls) == 0:
                inp = sentence
            else:
                if len(sentence_ls)+1 > dialogue_num:
                    inp = "|".join(sentence_ls[-dialogue_num+1:]+[sentence])
                else:
                    inp = "|".join(sentence_ls+[sentence])

        else:
            inp = sentence
        out = test_model(inp)
        print("唐小书：{}".format(out))
        sentence_ls.extend([sentence, out])

if __name__ == '__main__':
    main(long_dialogue=True)
