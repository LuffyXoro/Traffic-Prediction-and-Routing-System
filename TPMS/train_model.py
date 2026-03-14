import pandas as pd; import numpy as np; from sklearn.model_selection import train_test_split; from sklearn.preprocessing import MinMaxScaler; from tensorflow.keras.models import Sequential; from tensorflow.keras.layers import LSTM, Dense

data = pd.read_csv("traffic_data.csv")
data = data.select_dtypes(include=[np.number]).dropna()

scaler = MinMaxScaler(); data_scaled = scaler.fit_transform(data)

X = []; y = []; seq = 10
for i in range(len(data_scaled)-seq): X.append(data_scaled[i:i+seq]); y.append(data_scaled[i+seq])
X = np.array(X); y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,shuffle=False)

model = Sequential(); model.add(LSTM(64,return_sequences=True,input_shape=(X_train.shape[1],X_train.shape[2]))); model.add(LSTM(32)); model.add(Dense(y_train.shape[1])); model.compile(optimizer="adam",loss="mse")

model.fit(X_train,y_train,epochs=50,batch_size=32,validation_data=(X_test,y_test))

model.save("traffic_lstm_model.h5")
np.save("scaler.npy",scaler)
