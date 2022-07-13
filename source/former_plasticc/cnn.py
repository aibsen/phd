from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.utils import to_categorical
import h5py
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def load_data(filename):
    data = h5py.File(filename,'r')
    X = data['X'].value
    Y = data['Y'].value
    return [X,Y]

def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(39, 40,6)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(14, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer="sgd")
    return model

if __name__ == "__main__":
    #X,Y = load_data("dmdts_per_band.h5")
    #model = create_model()
    #print(Y.shape)
    #X=np.reshape(X,((7848,39, 40,6)))
    # Y=np.reshape(Y,(7848,14,1))
    #Ydf=pd.DataFrame(Y)
    #classes=Ydf.drop_duplicates().values
    #Y=to_categorical(Y, 14)
    #print(Y.shape)

    #x_train, x_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.20)
    #model.fit(x_train, y_train, batch_size=100, epochs=100, validation_data=(x_valid, y_valid),verbose=1, shuffle=True)
    data = h5py.File("dmdts_per_band.h5",'r')
    print(data)
	 
