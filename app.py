import tensorflow as tf
import numpy as np
import pickle
import gradio as gr
import os
import urllib.request

MODEL_URL = "https://huggingface.co/Newton-10/en-fr-transformer/resolve/main/transformer_model.keras"
VOCABS_URL = "https://huggingface.co/Newton-10/en-fr-transformer/resolve/main/vocabs.pkl"

if not os.path.exists('transformer_model.keras'):
    print("Downloading model...")
    urllib.request.urlretrieve(MODEL_URL, 'transformer_model.keras')

if not os.path.exists('vocabs.pkl'):
    print("Downloading vocabs...")
    urllib.request.urlretrieve(VOCABS_URL, 'vocabs.pkl')

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_embeddings = tf.keras.layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim, mask_zero=True
        )
        matrix = pos_enc_matrix(sequence_length, embed_dim)
        self.position_embeddings = tf.constant(matrix, dtype='float32')

    def call(self, inputs):
        embedded_tokens = self.token_embeddings(inputs)
        return embedded_tokens + self.position_embeddings

    def compute_mask(self, *args, **kwargs):
        return self.token_embeddings.compute_mask(*args, **kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({
            'sequence_length': self.sequence_length,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
        })
        return config

def pos_enc_matrix(L, d, n=10000):
    d2 = d // 2
    p = np.zeros((L, d))
    k = np.arange(L).reshape(-1, 1)
    i = np.arange(d2).reshape(1, -1)
    denom = np.power(n, -i / d2)
    args = k * denom
    p[:, 0::2] = np.sin(args)
    p[:, 1::2] = np.cos(args)
    return p

model = tf.keras.models.load_model(
    'transformer_model.keras',
    custom_objects={'PositionalEmbedding': PositionalEmbedding},
    compile=False
)

with open('vocabs.pkl', 'rb') as fp:
    data = pickle.load(fp)

eng_vect = tf.keras.layers.TextVectorization.from_config(data['eng_config'])
eng_vect.set_vocabulary(data['eng_vocab'])

fre_vect = tf.keras.layers.TextVectorization.from_config(data['fre_config'])
fre_vect.set_vocabulary(data['fre_vocab'])

lookup = list(fre_vect.get_vocabulary())
seq_length = 25
start_sent, end_sent = "[start]", "[end]"

def translate(sentence):
    enc_tokens = eng_vect([sentence])
    output_sent = [start_sent]

    for i in range(seq_length):
        vector = fre_vect([' '.join(output_sent)])
        dec_tokens = vector[:, :-1]
        pred = model([enc_tokens, dec_tokens])
        word = lookup[np.argmax(pred[0, i, :])]
        output_sent.append(word)
        if word == end_sent:
            break

    return ' '.join(output_sent[1:-1] if output_sent[-1] == end_sent else output_sent[1:])

demo = gr.Interface(
    fn=translate,
    inputs=gr.Textbox(label="English", placeholder="Type a sentence..."),
    outputs=gr.Textbox(label="French"),
    title="English → French Translator",
    description="Note: translations may be inaccurate — model uses greedy decoding without a causal mask fix (planned)."
)

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
