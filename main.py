import pandas 
import tensorflow 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.optimizers import SGD
opt = SGD(lr=0.01)





dataset = pandas.read_csv('/Users/oliverjohnson/loan-eligibility-predictor/loan-train.csv')
x = dataset.drop(columns=['Loan_Status'])
y = dataset['Loan_Status']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


model = tensorflow.keras.models.Sequential()

model.add(tensorflow.keras.Input(shape=(x_train_scaled.shape[1])))

#input layers
model.add(tensorflow.keras.layers.Dense(256, activation='sigmoid')) 
#hidden layers
model.add(tensorflow.keras.layers.Dense(128, activation='sigmoid')) 
model.add(tensorflow.keras.layers.Dense(64, activation='sigmoid')) 
model.add(tensorflow.keras.layers.Dense(32, activation='sigmoid')) 
model.add(tensorflow.keras.layers.Dense(16, activation='sigmoid')) 
model.add(tensorflow.keras.layers.Dense(8, activation='sigmoid')) 


#output layer
model.add(tensorflow.keras.layers.Dense(1, activation='relu'))
model.compile(loss = "binary_crossentropy", optimizer = "adam",metrics=['accuracy'])
model.fit(x_train_scaled,y_train, epochs=300, batch_size=32,validation_split=0.2)

model.evaluate(x_test_scaled,y_test)



