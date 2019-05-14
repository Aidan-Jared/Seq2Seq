# Seq2Seq

For this project I decided to look at [the Keras Seq2Seq](https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py) example to build my own seq2seq translation model

## Text Processing

In order to build the model the text first had to be split up and prossesed for the model to read and predict. The data set I used wat [this](http://www.manythings.org/anki/fra-eng.zip) which are english french pairs. The input and target data where on the same line seperated by a tab so each line was split at the tab and added to an input and target list. On top of this, the target text had a tab added to the front while a new line was added to the end. While all of this was happening, the input and target text were iterated through to find all the character used in both data sets.

With this done, the parameters for the input and output matrix where found baised on the number of input and output tokens, and the longest seq for the input and output. After this a dictionary was made of character and index pairs. Then three zero numpy arrays (encoder, decoder, and output) where made where they had the shape of (# of input text, max seq size, number of tokens). The final step was iterating through the input and output text and for each charachter adding a 1 to its location as determined by which line it was in, where in the line it was, and what the character index pair is.

With all of this done it was time to move onto builing the model

## Model