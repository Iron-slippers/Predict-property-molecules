import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
import plotly.graph_objs as go

from rdkit import Chem
from chemprop.features import morgan_binary_features_generator
from descriptastorus.descriptors import rdNormalizedDescriptors
from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# Data preprocessing
data = pd.read_excel('35000.xlsx')
data.describe()

data.drop(['Title', 'Unnamed: 0'], axis=1, inplace=True)

sns.histplot(data['IC50'])

data = data[data['IC50'] < 20].copy()


# Create functions for translate from structural form to set of features
# First descriptor
def create_descriptors(data_frame):
    rdkit_features = dict()
    rdkit_features['target'] = data_frame['IC50']

    features = [column[0] for column in generator.columns]
    for feature in features:
        rdkit_features[feature] = []

    for index, line in data_frame.iterrows():
        smiles = line['SMILES']
        values_of_features = rdkit_2d_features(smiles)

        for index, feature in enumerate(features):
            rdkit_features[feature].append(values_of_features[index])

    return pd.DataFrame(rdkit_features)


def rdkit_2d_features(smiles: str):
    features = generator.process(smiles)

    if features[0] == False:
        return None
    
    return features[1:]


# Second descriptor
def morgan_fingerprint(data_frame):
    molecules = list(map(Chem.MolFromSmiles, data_frame['SMILES']))
    descriptions = [morgan_binary_features_generator(molecule) for molecule in molecules]
    features = pd.DataFrame(descriptions)
    features['target'] = data_frame['IC50']
    return features


# First converted dataframe
generator = rdNormalizedDescriptors.RDKit2DNormalized()
data_descriptors = create_descriptors(data)
data_descriptors.dropna(inplace=True)
data_descriptors.shape


# Second converted dataframe
data_after_morgan = morgan_fingerprint(data)
data_after_morgan.dropna(inplace=True)
data_after_morgan.shape


# Use one of the two converted dataframes
y = np.array(data_after_morgan['target'])
x = np.array(data_after_morgan.drop('target', axis=1))


# Split to train and test
amount_features = x.shape[1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

x_train = tf.reshape(tf.cast(x_train, tf.float32), [-1, amount_features])
x_test = tf.reshape(tf.cast(x_test, tf.float32), [-1, amount_features])

y_train = tf.reshape(tf.cast(y_train, tf.float32), [-1, 1])
y_test = tf.reshape(tf.cast(y_test, tf.float32), [-1, 1])


# Create NN based on classes of Keras
model = Sequential()
model.add(Dense(amount_features // 2, activation='relu', input_shape=(amount_features,)))
model.add(Dense(amount_features // 2, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='rmsprop', loss='mse', metrics=['mape'])
history = model.fit(x_train, y_train, epochs=40, validation_split=0.1, verbose=2)


# Function for drow predict and target
def drow_result_of_model(x, y):

    predict_values = np.concatenate(model.predict(x))
    y = np.concatenate(y)
    size = np.arange(len(y))

    sorted_arrays = sorted(zip(y, predict_values))
    y, predict_values = zip(*sorted_arrays)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=size, y=predict_values, mode='markers', marker=dict(size=4), name='predict'))
    fig.add_trace(go.Scatter(x=size, y=y, line=dict(color='red', width=3), name='target'))
    fig.show()


drow_result_of_model(x_test, y_test)
