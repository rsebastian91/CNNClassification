# -*- coding: utf-8 -*-
"""

@author: robin
"""

class Classification():
    def __init__(self,x_train,y_train,x_valid,y_valid):
        self.x_train=x_train
        self.y_train=y_train 
        self.x_valid=x_valid 
        self.y_valid=y_valid



    # explore data
    def check_data(self):
    
        nb_train_sam=len(self.y_train)
        nb_valid_sam=len(self.y_valid)
    
    
        print("Training sample size: "+str(nb_train_sam))
        print("Validation sample size: "+str(nb_valid_sam))
    
        print("")
        print("Training data structure: "+str(self.x_train.shape))
        print("Training data type: "+str(self.x_train.dtype))
        print("Training data index structure: "+str(self.y_train.shape))
        print("Training data index type: "+str(self.y_train.dtype))
    
        print('')
        print("Validation data structure: "+str(self.x_valid.shape))
        print("Validation data type: "+str(self.x_valid.dtype))
        print("Validation data index structure: "+str(self.y_valid.shape))
        print("Validation data index type: "+str(self.y_valid.dtype))
    
        #print("")
        #print("Training data limits: Min="+str(self.x_train.min())+" Max="+str(self.x_train.max()))
    
    
    #data preparation
    def data_preparation(self,flaten,normalise):
        
        nb_train_sam=len(self.y_train)
        nb_valid_sam=len(self.y_valid)
        
        if (flaten==True):
            #Flatenning
            x_train_isize=self.x_train.shape[0]
            x_train_jsize=self.x_train.shape[1]
            x_train_ksize=self.x_train.shape[2]
        
            x_valid_isize=self.x_valid.shape[0]
            x_valid_jsize=self.x_valid.shape[1]
            x_valid_ksize=self.x_valid.shape[2]    
            
            self.x_train = self.x_train.reshape(nb_train_sam, x_train_jsize*x_train_ksize)
            self.x_valid = self.x_valid.reshape(nb_valid_sam, x_valid_jsize*x_valid_ksize)
            
            print("")
            print("Training data structure after flattening: "+str(self.x_train.shape))
            print("Validation data structure after flattening: "+str(self.x_valid.shape))
        
        if (normalise==True):
            #Normalising data
            self.x_train = self.x_train / self.x_train.max()
            self.x_valid = self.x_valid / self.x_train.max() 
            
            print("")
            print("Normalising data values between "+str(self.x_train.min())+" and "+str(self.x_train.max()))
            print("Data type changed to "+str(self.x_train.dtype))
    
    
    
    def data_encoding(self,encoding):
        if(encoding=='ordinal'):
            from sklearn.preprocessing import OrdinalEncoder
            oe = OrdinalEncoder()
            self.x_train = oe.fit_transform(self.x_train)
            self.x_valid = oe.fit_transform(self.x_valid)
            
        if(encoding=='onehot'):
            from sklearn.preprocessing import OrdinalEncoder
            oh = OrdinalEncoder()
            self.x_train = oh.fit_transform(self.x_train)
            self.x_valid = oh.fit_transform(self.x_valid)    
    
    #target encoding
    def target_encoding(self,num_categories,encoding):
            
            if(encoding=='binarymartix'):
                import tensorflow.keras as keras      
        
                print("Convert to binary class matrix")
                self.y_train = keras.utils.to_categorical(self.y_train, num_categories)
                self.y_valid = keras.utils.to_categorical(self.y_valid, num_categories)
                
            if(encoding=='label'):
                from sklearn.preprocessing import LabelEncoder
                labelencoder_Y = LabelEncoder()
                self.y_train = labelencoder_Y.fit_transform(self.y_train)
                self.y_valid = labelencoder_Y.fit_transform(self.y_valid)
    
    #plot accuracy and loss
    def plot_acc_and_loss(self,acc,val_acc,loss,val_loss):
        import matplotlib.pyplot as plt
        epochs = range(1, len(acc) + 1)
    
        plt.figure()
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.xlabel("Epochs")
        plt.ylabel("Percentage")
        plt.title('Training and validation accuracy')
        plt.legend()
    
        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.xlabel("Epochs")
        plt.title('Training and validation loss')
        plt.legend()        