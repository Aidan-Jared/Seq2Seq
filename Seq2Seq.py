from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np

def decode_seq(input_seq):
    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1,1, num_decoder_tokens))
    target_seq[0,0, target_token_index['\t']] = 1
    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0,-1,:])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char
        if sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq:
            stop_condition = True
        target_seq = np.zeros((1,1, num_decoder_tokens))
        target_seq[0,0,sampled_token_index] = 1
        states_value = [h, c]
    return decoded_sentence

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
    encoder_input_data = np.zeros((len(input_texts), max_encoder_seq, num_encoder_tokens), dtype='float32')
    decoder_input_data = np.zeros((len(target_texts), max_decoder_seq, num_decoder_tokens), dtype='float32')
    decoder_target_data = np.zeros((len(input_texts), max_decoder_seq, num_decoder_tokens), dtype='float32')

    # hot encoding input matrix
    for index, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for j, char in enumerate(input_text):
            encoder_input_data[index, j, input_token_index[char]] = 1
        for j, char in enumerate(target_text):
            decoder_input_data[index, j, target_token_index[char]] = 1
            if j > 0:
                decoder_target_data[index, j - 1, target_token_index[char]] = 1

    # Building the encoder
    encoder_input = Input(shape=(None, num_encoder_tokens))
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_input)
    encoder_states = [state_h, state_c]

    # Building the Decoder
    decoder_input = Input(shape=(None, num_decoder_tokens))
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_input, initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # training the model
    model = Model([encoder_input, decoder_input], decoder_outputs)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=batch_size, epochs=epochs, validation_split=.2)
    model.save('s2s.h5')

    # sampling model
    encoder_model = Model(encoder_input, encoder_states)

    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_input, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_input] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
    reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

    for seq_index in range(100):
        input_seq = encoder_input_data[seq_index: seq_index + 1]
        sentence = decode_seq(input_seq)
        print('-')
        print('Input sentence: ', input_texts[seq_index])
        print('Decoded sentence: ', sentence)