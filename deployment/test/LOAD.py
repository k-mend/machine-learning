#!/usr/bin/env python
# coding: utf-8

# In[3]:

import pickle

from flask import Flask
from flask import request
from flask import jsonify

# In[4]:


with open('model_C=1.0.bin', 'rb') as f_in:
    dv,model = pickle.load(f_in)


# In[5]:


test_dicts = {'customerid': '4183-myfrb',
             'gender': 'female',
             'seniorcitizen': 0,
             'partner': 'no',
             'dependents': 'no',
             'tenure': 21,
             'phoneservice': 'yes',
             'multiplelines': 'no',
             'internetservice': 'fiber_optic',
             'onlinesecurity': 'no',
             'onlinebackup': 'yes',
             'deviceprotection': 'yes',
             'techsupport': 'no',
             'streamingtv': 'no',
             'streamingmovies': 'yes',
             'contract': 'month-to-month',
             'paperlessbilling': 'yes',
             'paymentmethod': 'electronic_check',
             'monthlycharges': 90.05,
             'totalcharges': '1862.9',
             'churn': 0}


# In[7]:


x_test = dv.transform(test_dicts)


# In[9]:


model.predict_proba(x_test)[:,1]


# In[10]:
app = Flask("predict")


# In[14]:


@app.route('/predict', methods = ['POST'])
def predict():
    #json is same as python dictionary only that it has double quotes and python has single
    customer = request.get_json() #converts customer details to json
    X_customer = dv.transform(customer)
    y_pred = model.predict_proba(X_customer)[0,1]
    churn = y_pred>0.5

    result = {
        'churn_prediction' : float(y_pred),  #add the conversions otherwise it will raise an error
        'churn' : bool(churn)
    }
    return result

if __name__=="__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)

    


# In[ ]:




