import pandas 
import tensorflow 
from sklearn.model_selection import train_test_split


dataset = pandas.read_csv('/Users/oliverjohnson/loan-eligibility-predictor/loan-train.csv')
x = dataset.drop(columns=['Loan_Status'])
y = dataset['Loan_Status']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01)

model = tensorflow.keras.models.Sequential()

model.add(tensorflow.keras.Input(shape=(x_train.shape)))

#input layers
model.add(tensorflow.keras.layers.Dense(256, activation='sigmoid')) 
#hidden layers
model.add(tensorflow.keras.layers.Dense(256, activation='sigmoid')) 
#output layer
model.add(tensorflow.keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train, epochs=1000)




