import pandas as pd
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences


# IGN Dataset loading
df = pd.read_csv("ign.csv")

print(df.head())

# Our labels will be generated starting from data of column 'score_phrase'
print(df['score_phrase'].unique())
# To generate a baseline for the classification we convert them to positive and
# negative
positive = ['Masterpiece', 'Amazing', 'Great', 'Good', 'Okay']
negative = ['Awful', 'Mediocre', 'Bad', 'Painful', 'Unbearable', 'Disaster']

def convert_to_binary(val):
    if val in positive:
        return 'positive'
    elif val in negative:
        return 'negative'
    else:
        return val


df['binary'] = df['score_phrase'].apply(convert_to_binary)

print(df['binary'].unique())

print(df.columns)
# We don't have a full review but we can combine several columns to get one.
# We assume url and release date are not relevant for the final score
print(df['editors_choice'].unique())
# 'editors_choice' is composed by Y and N only
print(df.loc[df['editors_choice'] == 'Y', 'binary'].value_counts())
print(df.loc[df['editors_choice'] == 'N', 'binary'].value_counts())
# We can see that such feature is quite important since whenever the game
# is an editor's choice it gets always a positive review. Since we want to
# perform sentiment analysis on reviwews it is better to convert 'Y' in something
# more clear like 'editors_choice'
df['editors_choice'] = df['editors_choice'].map({'Y' : 'editors_choice', 'N': ''})

to_merge = ['title', 'platform', 'genre', 'editors_choice']

df['predictors'] = df['title']
for col in to_merge[1:]:
    def insert_space(val):
        if pd.isnull(val):
            return ' '
        else:
            return val + ' '

    df['predictors'] += df[col].apply(insert_space)

#print(df[['predictors', 'title', 'platform', 'genre', 'editors_choice']].head())

# Now we can create our test and training set
from sklearn.model_selection import train_test_split
X = df['predictors']
y = df['binary']

trainX, testX, trainY, testY = train_test_split(X, y,
test_size=0.33, random_state=1)


raise Exception()

# Data preprocessing
# Sequence padding
trainX = pad_sequences(trainX, maxlen=100, value=0.)
testX = pad_sequences(testX, maxlen=100, value=0.)
# Converting labels to binary vectors
trainY = to_categorical(trainY, nb_classes=2)
testY = to_categorical(testY, nb_classes=2)

# Network building
net = tflearn.input_data([None, 100])
net = tflearn.embedding(net, input_dim=10000, output_dim=128)
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
          batch_size=32)
