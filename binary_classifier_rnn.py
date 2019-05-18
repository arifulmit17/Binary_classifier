import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotting_data as pltdt
import pred_and_evaluation as pred_eva


def binary_classifier(train_data):
 
        Classes=train_data.Class
        for i,Class in enumerate(Classes):
           if Class!='F':
             train_data.Class[i]='NF'
        train_data.to_csv('train_binc_2.csv',index = False)
        print ('Types of Unique Classes',train_data.Class.unique())
        plt.figure(figsize=(4,4))
        train_data.Class.value_counts().plot(kind='bar');
        plt.show()
        train_data=pd.read_csv('train_binc_2.csv')
        text_1=train_data.RequirementText_string
        
        dic={'F':0,'NF':1}
        
        length_data=len(train_data.index)
        print ('Number of total training sentences',length_data)
        
        num_labels=len(train_data.Class)
        print ('Number of total training labels',num_labels)
        for i in range(length_data):
           if i in train_data.index:
             d=train_data.Class[i]
             train_data.Class[i]=dic[d]
        
        print ('Before dropping validation data: ',train_data.shape)
        val_data=train_data.sample(frac=0.1)
        print('Validation data sample :',val_data.head(3))
        train_data=train_data.drop(val_data.index)
        print ('After dropping validation data: ',train_data.shape)
        test_data=train_data.sample(frac=0.10)
        print('Test data sample :',test_data.head(3))
        print('Training data shape:',train_data.shape)
        print('validation data shape:',val_data.shape)
        print('Testing data shape:',test_data.shape)
       
        print(train_data.isnull().sum())
        print(val_data.isnull().sum())
        print(test_data.isnull().sum())
        train_data=train_data.drop(test_data.index)
        print ('After dropping test data: ',train_data.shape)
#################################################### Tokenize both the dataset ####################################################
        def sent_tokin(train_data,num_words):
             from keras.preprocessing.text import Tokenizer

             texts=train_data.RequirementText_string
             NUM_WORDS=num_words
             tokenizer = Tokenizer(num_words=NUM_WORDS,lower=True)
             tokenizer.fit_on_texts(texts)
             sequences_train = tokenizer.texts_to_sequences(texts)
             sequences_valid=tokenizer.texts_to_sequences(val_data.RequirementText_string)
             word_index = tokenizer.word_index
             return tokenizer,texts,word_index,sequences_train,sequences_valid



        NUM_WORDS=9000
        tokenizer,texts,word_index,sequences_train,sequences_valid=sent_tokin(train_data,NUM_WORDS)
        print('Found %s unique tokens.' % len(word_index))
        
        
        ############################################### Zero padding the dataset ##################################################
        y_train = train_data.Class.astype(np.int64) 
        
        
        
        length=len(sequences_train)
        lenth=[]
        for i in range(length):
           lenth.append(len(sequences_train[i])) 
        max_length=61 #np.max(lenth)
        print ('maxlength :',max_length)
        
        print ('First sentence: ',texts[0])
        print ('First sentence after tokenizing: ',sequences_train[0])
        
        for i in range(length):
         if (len(sequences_train[i]) < max_length):
             for j in range(max_length-len(sequences_train[i])):
                sequences_train[i].append(0)
        sequences_train=np.array(sequences_train)
       
       
       
       
        print ('First sentence after tokenizing and zero padding: ',sequences_train[0])
        print ('train_data.index', train_data.index)
        
        X_train = sequences_train
        print ('Training data shape:',X_train.shape)
        print ('First sentence after tokenizing and zero padding: ',X_train[0])
        print ('Length of First sentence after tokenizing and zero padding:',len(X_train[0]))
        
        length=len(sequences_valid)
        print(length)
        for i in range(length):
         if (len(sequences_valid[i]) < max_length):
             for j in range(max_length-len(sequences_valid[i])):
                sequences_valid[i].append(0)
      
        sequences_valid=np.array(sequences_valid)
        
        X_val=sequences_valid
        print ('validation data sequence ',X_val[0])
        
        
        
        y_val=val_data.Class.astype(np.int64)
        print('Shape of X train and X validation data:', X_train.shape,X_val.shape)
        print('Shape of label train and validation data:', y_train.shape,y_val.shape)
       


        sequences_test=tokenizer.texts_to_sequences(test_data.RequirementText_string)
        length=len(sequences_test)
        print (length)
       
       
        for i in range(length):
         if (len(sequences_test[i]) < max_length):
             for j in range(max_length-len(sequences_test[i])):
                sequences_test[i].append(0)
       
        sequences_test=np.array(sequences_test)
        print ('First sentence of testing dataset after tokenizing and zero padding: ',sequences_test[0])
       
        print ('Testing data shape:',len(sequences_test))
        X_test=sequences_test
        
        
        y_test=test_data.Class.astype(np.int64)
        print (X_test.shape)
        print (y_test.shape)

########################################### Word2vec on the dataset ####################################################
        def word2vec_embed(text_1,word_index,NUM_WORDS,embedding_dim):
              
              import nltk
              from gensim.models import Word2Vec
              from gensim.models.keyedvectors import KeyedVectors
              EMBEDDING_DIM = embedding_dim 
             
              word_vectors = KeyedVectors.load_word2vec_format('C:\\Users\\User\\Documents\\resources\\Binary_classifier\\GoogleNews-vectors-negative300.bin', binary=True)
              vocabulary_size=min(len(word_index)+1,NUM_WORDS)
              print ('vocabulary size:',vocabulary_size)
              print ('Num_words: ',NUM_WORDS)
              print ('length_word_index:' ,len(word_index))
              
              embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))
              for word, i in word_index.items():
                 if i < vocabulary_size:
                  if word in word_vectors.vocab:
                    embedding_matrix[i] = word_vectors.word_vec(word)
                    
                
                
              print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
              
             

            
              return vocabulary_size,embedding_matrix
#################################################### Model Building ####################################################
        EMBEDDING_DIM= 300
        vocabulary_size,embedding_matrix=word2vec_embed(text_1,word_index,NUM_WORDS,EMBEDDING_DIM)
        from keras.layers import  Dense, Embedding,GRU
        from keras.optimizers import Adam
        from keras.models import Sequential
        
        sequence_length = X_train.shape[1]
       


        model = Sequential()
        model.add(Embedding(vocabulary_size,EMBEDDING_DIM,weights=[embedding_matrix], input_length=sequence_length,mask_zero=True ,trainable=False))
        model.add(GRU(units=300,dropout=0.2,recurrent_dropout=0.2))
        model.add(Dense(units=1,activation='sigmoid'))
        
        
        
        print (model.summary())
        adam = Adam(lr=0.001,beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['acc'])
#################################################### Training the model ####################################################
        
        history = model.fit(X_train, y_train, batch_size=50, epochs=30, verbose=1, validation_data=(X_val, y_val))  # starts training
        history_dict = history.history
        print(history_dict.keys())
        
 
        
#################################################### Predicting the Test dataset and evaluation ################################################### 
        pred_eva.pred_and_evaluation(model,X_test,y_test,X_val, y_val,dic)
#################################################### Plot the evaluation ####################################################
        pltdt.plotting_data(history_dict)
