from flask import Flask,jsonify,request
from sklearn.externals import joblib
from flask_cors import CORS
import numpy as np

app=Flask(__name__)

CORS(app)

def predict(x):
    clf=joblib.load('model.pkl')
  
#   
    a=x['clumpthickness']
    b=x['cellsize']
    c=x['cellshape']
    d=x['marginal']
    e=x['epithelial_cell_size']
    f=x['bare_nuclei']
    g=x['bland_chromatin']
    h=x['normal_nucleoli']
    i=x['mitoses']

    arr=np.array([a,b,c,d,e,f,g,h,i])
    z=clf.predict(arr.reshape(1,-1))
    
    return int(z)

@app.route('/predict',methods=['POST'])
def home(): 
    
    data= request.get_json()
    print(data)
    y=predict(data)
    return jsonify({'ans':y})


app.run(port=8090)