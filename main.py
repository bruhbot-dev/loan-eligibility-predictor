import pandas 
import tensorflow 

dataset = pandas.read_csv('/Users/oliverjohnson/loan-eligibility-predictor/loan-train.csv')
x = dataset.drop(columns=['Loan_Status'])
y = dataset['Loan_Status']

model = tensorflow.keras.models.Sequential()

model.add(tensorflow.keras.Input(shape=(x.shape)))

#input layers
model.add(tensorflow.keras.layers.Dense(256, activation='sigmoid')) 
#hidden layers
model.add(tensorflow.keras.layers.Dense(256, activation='sigmoid')) 
#output layer
model.add(tensorflow.keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

model.fit(x,y, epochs=1000)




