import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
def pred_and_evaluation(model,X_test,y_test,X_val, y_val,dic):
              
              y_pred_validation=model.predict_classes(X_val)
              y_test_2=[]
              y_test_2=np.asarray(y_val)
              print (dic)
              print ('Actual validation values =',y_test_2)
              print ('Predicted values= ',y_pred_validation.flatten())
              my_tags= ['F','NF']
              print('Validation accuracy is %s' % accuracy_score(y_test_2,y_pred_validation))
              print('Validation performance',classification_report( y_test_2 ,y_pred_validation ,target_names=my_tags))
              score = model.evaluate(X_val, y_val, verbose=1)
              cm = confusion_matrix(y_test_2,y_pred_validation)
              cm_df=pd.DataFrame(cm,index=['F','NF'],columns=['F','NF'])
              print (cm_df)
              
         
              y_pred_test=model.predict_classes(X_test)
              y_test_2=[]
              y_test_2=np.asarray(y_test)
              print (dic)
              print ('Actual test values =',y_test_2)
              print ('Predicted values= ',y_pred_test.flatten())
              my_tags= ['F','NF']
              print('Testing accuracy is %s' % accuracy_score(y_test_2,y_pred_test))
              print('Testing performance',classification_report( y_test_2 ,y_pred_test ,target_names=my_tags))
              score = model.evaluate(X_val, y_val, verbose=1)
              cm = confusion_matrix(y_test_2,y_pred_test)
              cm_df=pd.DataFrame(cm,index=['F','NF'],columns=['F','NF'])
              print (cm_df)
              print('Validation loss:', score[0])
              print('Validation accuracy:', score[1])
    
    
         