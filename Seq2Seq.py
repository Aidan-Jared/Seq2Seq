from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np

if __name__ == "__main__":
    # setting parameters
    batch_size = 64
    epochs = 100
    latent_dim = 256 # size of the encoding space
    num_samples = 10000
    data = 'fra.txt'
    input_texts = []
    target_texts = []
    input_chars = set()
    target_chars = set()

    # reading and processing text
    with open(data, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    for i in lines[:min(num_samples, len(lines) - 1)]:
        input_text, target_text = i.split('\t')
        target_text = '\t' + target_text + '\n'
        input_texts.append(input_text)
        target_texts.append(target_text)
        for j in input_text:
            if j not in input_chars:
                input_chars.add(j)
        for j in target_text:
            if j not in target_chars:
                target_chars.add(j)
    
    # finding sizes for input matrix
    input_chars = sorted(list(input_chars))
    target_chars = sorted(list(target_chars))
    num_encoder_tokens = len(input_chars)
    num_decoder_tokens = len(target_chars)
    max_encoder_seq = max([len(text) for text in input_texts])
    max_decoder_seq = max([len(text) for text in target_texts])

    # creating index char pairs for input and target
    input_token_index = dict([(i, index) for index, i in enumerate(input_chars)])
    target_token_index = dict([(i, index) for index, i in enumerate(target_chars)])

    # creating empty input matrix
    encoder_input = np.zeros((len(input_texts), max_encoder_seq, num_encoder_tokens), dtype='float32')
    decoder_input = np.zeros((len(target_texts), max_decoder_seq, num_decoder_tokens), dtype='float32')
    decoder_target = np.zeros((len(input_texts), max_decoder_seq, num_decoder_tokens), dtype='float32')

    # hot encoding input matrix
    for index, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for j, char in enumerate(input_text):
            encoder_input[index, j, input_token_index[char]] = 1
        for j, char in enumerate(target_text):
            decoder_input[index, j, target_token_index[char]] = 1
            if j > 0:
                decoder_target[index, j - 1, target_token_index[char]] = 1