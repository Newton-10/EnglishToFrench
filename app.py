import numpy as np
import pickle
import gradio as gr
import os
import re
import string
import urllib.request

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite import Interpreter  # fallback if tflite_runtime unavailable

MODEL_URL = "https://huggingface.co/Newton-10/en-fr-transformer/resolve/main/transformer_model%20(2).tflite"
VOCABS_URL = "https://huggingface.co/Newton-10/en-fr-transformer/resolve/main/vocabs.pkl"

if not os.path.exists('transformer_model.tflite'):
    urllib.request.urlretrieve(MODEL_URL, 'transformer_model.tflite')
if not os.path.exists('vocabs.pkl'):
    urllib.request.urlretrieve(VOCABS_URL, 'vocabs.pkl')

with open('vocabs.pkl', 'rb') as fp:
    data = pickle.load(fp)

eng_vocab = data['eng_vocab']
fre_vocab = data['fre_vocab']
eng_word_to_id = {w: i for i, w in enumerate(eng_vocab)}
fre_word_to_id = {w: i for i, w in enumerate(fre_vocab)}
fre_id_to_word = {i: w for i, w in enumerate(fre_vocab)}

SEQ_LENGTH = 25
PUNCT_RE = re.compile('[%s]' % re.escape(string.punctuation))

def standardize(text):
    text = text.lower()
    text = PUNCT_RE.sub('', text)
    return text

def vectorize(text, word_to_id, length):
    tokens = standardize(text).split()
    ids = [word_to_id.get(t, 1) for t in tokens]  # 1 = [UNK]
    ids = ids[:length]
    ids = ids + [0] * (length - len(ids))  # 0 = padding
    return np.array([ids], dtype=np.float32)

interpreter = Interpreter(model_path='transformer_model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

enc_idx = next(d['index'] for d in input_details if 'encoder' in d['name'])
dec_idx = next(d['index'] for d in input_details if 'decoder' in d['name'])
out_idx = output_details[0]['index']

start_sent, end_sent = "[start]", "[end]"

def translate(sentence):
    enc_tokens = vectorize(sentence, eng_word_to_id, SEQ_LENGTH)
    output_sent = [start_sent]

    for i in range(SEQ_LENGTH):
        dec_tokens = vectorize(' '.join(output_sent), fre_word_to_id, SEQ_LENGTH)

        interpreter.set_tensor(enc_idx, enc_tokens)
        interpreter.set_tensor(dec_idx, dec_tokens)
        interpreter.invoke()
        pred = interpreter.get_tensor(out_idx)

        word_id = int(np.argmax(pred[0, i, :]))
        word = fre_id_to_word.get(word_id, "")
        output_sent.append(word)
        if word == end_sent:
            break

    return ' '.join(output_sent[1:-1] if output_sent[-1] == end_sent else output_sent[1:])

demo = gr.Interface(
    fn=translate,
    inputs=gr.Textbox(label="English", placeholder="Type a sentence..."),
    outputs=gr.Textbox(label="French"),
    title="English → French Translator",
    description="Note: translations may be inaccurate — model trained without a causal mask fix (planned)."
)

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
