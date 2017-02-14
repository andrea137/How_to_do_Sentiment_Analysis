# How_to_do_Sentiment_Analysis
This is my solution for the challenge:

'How to Do Sentiment Analysis' #3 - Intro to Deep Learning by Siraj Raval on Youtube (https://youtu.be/si8zZHkufRY)


##Overview

The challenge for this video is to train a model on [this](https://www.kaggle.com/egrinstein/20-years-of-games) dataset of video game reviews from IGN.com. Then, given some new video game title it should be able to classify it.

The code uses [pandas](http://pandas.pydata.org/) to parse the dataset. It uses [TFLearn](http://tflearn.org/) to train a Sentiment Analyzer and The neural network that is built for this is a [recurrent network](https://en.wikipedia.org/wiki/Recurrent_neural_network). It uses a technique called [LSTM](http://colah.github.io/posts/2015-08-Understanding-LSTMs/).

Each review has a label that's either 'Amazing', 'Great', 'Good', 'Awful', 'Okay', 'Mediocre', 'Bad', 'Painful', 'Unbearable', 'Disaster', or 'Masterpiece'. These are the emotions. 

The baseline is obtained by converting the labels to only 2 emotions (positive or negative).


##Usage

Run ``games_classification.ipynb`` in a jupyter notebeook



##Credits

Credits for the inital code (demo.py) go to the [author](https://github.com/aymericdamien) of TFLearn.
