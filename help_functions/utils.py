from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef, recall_score, precision_score
from keras.models import Sequential 
from keras.layers import Dense, Dropout
import pandas as pd

def NeuralNet2L(kernel_initializer='glorot_uniform', l1_neuron = 10, l2_neuron = 5, metrics = ['accuracy'], loss = 'binary_crossentropy', optimizer = 'adam', dropout = 0.2):
	nn_model = Sequential()  
	nn_model.add(Dense(l1_neuron, activation='relu',kernel_initializer=kernel_initializer))
	nn_model.add(Dropout(dropout))
	nn_model.add(Dense(l2_neuron, activation='relu',kernel_initializer=kernel_initializer))  
	nn_model.add(Dense(1, activation='sigmoid',kernel_initializer=kernel_initializer))    

	# compile model     
	nn_model.compile(loss=loss, optimizer=optimizer, metrics=metrics) 

	return nn_model
	
def NeuralNet1L(kernel_initializer='glorot_uniform', l1_neuron = 5, metrics = ['accuracy'], loss = 'binary_crossentropy', optimizer = 'adam'):
	nn_model = Sequential()  
	nn_model.add(Dense(l1_neuron, activation='relu',kernel_initializer=kernel_initializer))
	nn_model.add(Dense(1, activation='sigmoid',kernel_initializer=kernel_initializer))    

	# compile model     
	nn_model.compile(loss=loss, optimizer=optimizer, metrics=metrics) 

	return nn_model

def get_metrics(model, y_true, y_pred, time_to_train):
	FP = confusion_matrix(y_true, y_pred)[0,1] 
	FN = confusion_matrix(y_true, y_pred)[1,0]
	TP = confusion_matrix(y_true, y_pred)[1,1]
	TN = confusion_matrix(y_true, y_pred)[0,0]

	# Matthews correlation coefficient
	mcc = matthews_corrcoef(y_true, y_pred)

	# accuracy
	acc = accuracy_score(y_true, y_pred)

	# recall
	rec = recall_score(y_true, y_pred)

	# precision
	prec = precision_score(y_true, y_pred)

	# False negative rate
	FNR = FN/(TP+FN)

	# False alarm rate
	FAR = FP/(TP+FP)

	lst = [model, acc, rec, prec, FNR, FAR, mcc, time_to_train]

	return pd.DataFrame([lst], columns = ['model','accuracy', 'precision', 'recall','FNR','FAR','mcc','time_to_train'])