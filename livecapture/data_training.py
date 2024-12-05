import os  
import numpy as np 
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense 
from keras.models import Model

# Initialization
is_init = False
label = []
dictionary = {}
c = 0

# Load data from .npy files
for i in os.listdir():
    if i.endswith(".npy") and i != "labels.npy":  
        data = np.load(i)
        if data.size == 0:
            raise ValueError(f"File {i} is empty.")
        
        size = data.shape[0]
        labels = np.array([i.split('.')[0]] * size).reshape(-1, 1)

        if not is_init:
            is_init = True
            X = data
            y = labels
        else:
            X = np.concatenate((X, data))
            y = np.concatenate((y, labels))

        label.append(i.split('.')[0])
        dictionary[i.split('.')[0]] = c  
        c += 1

# Encode labels
for i in range(y.shape[0]):
    y[i, 0] = dictionary[y[i, 0]]
y = np.array(y, dtype="int32")
y = to_categorical(y)

# Shuffle data
shuffled_indices = np.random.permutation(X.shape[0])
X = X[shuffled_indices]
y = y[shuffled_indices]

# Validate data shape
if len(X.shape) != 2:
    raise ValueError(f"Expected 2D input, but got shape {X.shape}. Check your .npy files.")

# Build the model
ip = Input(shape=(X.shape[1],))
m = Dense(512, activation="relu")(ip)
m = Dense(256, activation="relu")(m)
op = Dense(y.shape[1], activation="softmax")(m) 

model = Model(inputs=ip, outputs=op)
model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])

# Train the model
model.fit(X, y, epochs=50)

# Save the model and labels
model.save("model.h5")
np.save("labels.npy", np.array(label))
print("Model and labels saved successfully!")
