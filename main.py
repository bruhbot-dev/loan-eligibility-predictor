import pandas 
import tensorflow 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Read the dataset from CSV
dataset = pandas.read_csv('/Users/oliverjohnson/loan-eligibility-predictor/loan-train.csv')

# Separate features (x) and target variable (y)
x = dataset.drop(columns=['Loan_Status'])
y = dataset['Loan_Status']

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Standardize the features 
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Build a neural network model 
model = tensorflow.keras.models.Sequential()
#nput layer 
model.add(tensorflow.keras.Input(shape=(x_train_scaled.shape[1])))
# output layer
model.add(tensorflow.keras.layers.Dense(1, activation='sigmoid')) 

# Compile the model 
model.compile(loss = "binary_crossentropy", optimizer = "adam",metrics=['accuracy'])
# Train the model
model.fit(x_train_scaled,y_train, epochs=1000)
# evaluate model based on test data
model.evaluate(x_test_scaled,y_test)



