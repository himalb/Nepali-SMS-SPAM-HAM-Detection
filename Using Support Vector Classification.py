
# coding: utf-8

# In[1]:


import pandas as pd
import re
from sklearn import svm


# In[2]:


sms=pd.read_csv("msgs.csv",encoding='utf-8')
sms=sms.rename(columns = {'c1':'label','c2':'message'})



X_train=sms["message"][:120]
Y_train=sms["label"][:120]
X_test=sms["message"][121:178]
Y_test=sms["label"][121:178]




import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer(analyzer = "word",tokenizer = None,preprocessor = None,stop_words = None,max_features = 5000)



train_data_features=vectorizer.fit_transform(X_train)
train_data_features=train_data_features.toarray()

test_data_features=vectorizer.transform(X_test)
test_data_features=test_data_features.toarray()

#SVM with linear kernel
clf=svm.SVC(kernel='linear',C=1.0)
print ("Training")
clf.fit(train_data_features,Y_train)




print ("Testing")
predicted=clf.predict(test_data_features)
accuracy=np.mean(predicted==Y_test)
print ("Accuracy: ",accuracy)

#Validation
X=sms["message"]
validation_data=vectorizer.transform(X)
validation_data=validation_data.toarray()




classification=clf.predict(validation_data)
for i in range(0,len(sms.message)):
    print(sms.message[i], classification[i])

