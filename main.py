import pandas 
import tensorflow 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler






dataset = pandas.read_csv('/Users/oliverjohnson/loan-eligibility-predictor/loan-train.csv')
x = dataset.drop(columns=['Loan_Status'])
y = dataset['Loan_Status']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


model = tensorflow.keras.models.Sequential()

model.add(tensorflow.keras.Input(shape=(x_train_scaled.shape[1])))

model.add(tensorflow.keras.layers.Dense(1, activation='sigmoid')) 




model.compile(loss = "binary_crossentropy", optimizer = "adam",metrics=['accuracy'])
model.fit(x_train_scaled,y_train, epochs=1000)

model.evaluate(x_test_scaled,y_test)



