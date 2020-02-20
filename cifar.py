# -*- coding: utf-8 -*-

import keras
import numpy as np
from keras.datasets import cifar10
from keras import optimizers
from keras.models import load_model
from keras import models
from keras import layers


#Load Data
(train_data,train_labels),(test_data,test_labels)=cifar10.load_data()

#Selecting subset of three classes
def subset_three_class(process_array_data,process_array_labels):
    x_data=[]
    y_data=[]
    for i in range(len(process_array_labels)):
        if process_array_labels[i]==0 or process_array_labels[i]==1 or process_array_labels[i]==2:
            x_data.append((process_array_data[i]))
            y_data.append(int(process_array_labels[i]))                
    return np.asarray(x_data),np.asarray(y_data)


x_train,y_train=subset_three_class(train_data,train_labels)
x_test,y_test=subset_three_class(test_data,test_labels)
#test=list(x_train) 
#print(x_train[0][0])  

#Vectorization
def vectorize_seq(seq,dim=3*32*32):
    trans_arr=[]
    for i,seque in enumerate(seq):
        trans_arr.append((seque.flatten())/255)        
    return np.asarray(trans_arr)

#Vectorization
x_train_f=vectorize_seq(x_train)
x_test_f=vectorize_seq(x_test)

#Categorical Encoding
from keras.utils.np_utils import to_categorical
one_hot_train_labels=to_categorical(y_train) 
one_hot_test_labels=to_categorical(y_test)   

#Spliting Training and Validation
part_x_train=x_train_f[:12000] 
x_val=x_train_f[12000:]

part_y_train=one_hot_train_labels[:12000]
y_val=one_hot_train_labels[12000:]



#Model Design
model=models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(3072,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(3,activation='sigmoid'))
rmsprop=keras.optimizers.RMSprop(learning_rate=0.0001, rho=0.9)
#Compile model
model.compile(optimizer=rmsprop,loss='binary_crossentropy',metrics=['accuracy'])


#-----------------------------------------------------------------
history=model.fit(part_x_train,part_y_train,epochs=50
                  ,batch_size=512,
                  validation_data=(x_val,y_val))

history_dict=history.history
model.save("model_cifar.hdf5")
print("Model Saved")

#Plot
import matplotlib.pyplot as plt
loss_values=history_dict['loss']
val_loss_values=history_dict['val_loss']
epochs=range(1,len(loss_values)+1)

#Loss Plot
plt.plot(epochs,loss_values,'bo',label='Training Loss')
plt.plot(epochs,val_loss_values,'b',label='Validation Loss')
plt.title('Traning and Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

#Accuracy Plot
plt.clf()
acc_values=history_dict['accuracy']
val_acc_values=history_dict['val_accuracy']
plt.plot(epochs,acc_values,'bo',label='Training ACcuy')
plt.plot(epochs,val_acc_values,'b',label='Validation Accuracy')
plt.title('Traning and Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

result=model.evaluate(x_val,y_val)
print(result)


#Load Saved Model and Evalulate it on Test Data
print("Loading Saved Model")
model_new=load_model("model_cifar.hdf5")

#Evaluating Model on test Data.
result=model_new.evaluate(x_test_f,one_hot_test_labels)


print("Test Results")
print(result)
