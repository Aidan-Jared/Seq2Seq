# Seq2Seq

For this project I decided to look at [the Keras Seq2Seq](https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py) example to build my own seq2seq translation model

## Text Processing

In order to build the model the text first had to be split up and prossesed for the model to read and predict. The data set I used wat [this](http://www.manythings.org/anki/fra-eng.zip) which are english french pairs. The input and target data where on the same line seperated by a tab so each line was split at the tab and added to an input and target list. On top of this, the target text had a tab added to the front while a new line was added to the end. While all of this was happening, the input and target text were iterated through to find all the character used in both data sets.

With this done, the parameters for the input and output matrix where found baised on the number of input and output tokens, and the longest seq for the input and output. After this a dictionary was made of character and index pairs. Then three zero numpy arrays (encoder, decoder, and output) where made where they had the shape of (# of input text, max seq size, number of tokens). The final step was iterating through the input and output text and for each charachter adding a 1 to its location as determined by which line it was in, where in the line it was, and what the character index pair is.

With all of this done it was time to move onto builing the model

## Model

The model is structured into three parts, the encoder, the decoder and the dense activation layer. The encoder and decoder are LSTM layers with a size of the lattent_dim which in the case of the keras tutorial is 256. The encoder takes as an input the hot encoded english text and puts out an output and the states of the encoder. The encoder output is ingnored and the encoder states are given to the decoder. The encoder state is the hidden state output (state_h) and cell state (state_c) of the last input while the return sequences is the hidden state output for all inputs. Refer to [here](https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/) for examples.

The decoder takes in the hot encoded french text but also is given the intial state from the encoder. The decoder then takes its ouput into a softmax dense layer the size of the number of tokens and trys to predict the end of the sequence.

After training the model needs to be sampled to predict on new sequences. In order to do this, an encoder model is made which takes in an encoder input and then the trained encoder states. This model then outputs the states_values which is the inputed sequence encoded.

A decoder model is also made which uses the trained decoder lstm layer and dense layer and takes in the start of sequence token and the encoded sequence. Then while the sequence is less than the max sequence length or the end token has not been predicted, the model predicts the next token baised off of the previously prdicted token.