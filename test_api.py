from utils import read_digits,preprocess_data
from api.app import app
import flask

def get_processed_data():
    X,y = read_digits()
    X = X[:100,:,:]
    y = y[:100]
    X = preprocess_data(X)
    return X,y
#main test cases for asseting digits prediction
def test_predict_0():
    X,y = get_processed_data()
    response = app.test_client().post("/predict", json={"image":X[0].tolist()})
    assert response.status_code == 200    
    assert response.get_json()['prediction'] == [0]

def test_predict_1():
    X,y = get_processed_data()
    response = app.test_client().post("/predict", json={"image":X[1].tolist()})
    assert response.status_code == 200    
    assert response.get_json()['prediction'] == [1]

def test_predict_2():
    X,y = get_processed_data()
    response = app.test_client().post("/predict", json={"image":X[2].tolist()})
    assert response.status_code == 200    
    assert response.get_json()['prediction'] == [2]

def test_predict_3():
    X,y = get_processed_data()
    response = app.test_client().post("/predict", json={"image":X[3].tolist()})
    assert response.status_code == 200    
    assert response.get_json()['prediction'] == [3]
    
def test_predict_4():
    X,y = get_processed_data()
    response = app.test_client().post("/predict", json={"image":X[4].tolist()})
    assert response.status_code == 200    
    assert response.get_json()['prediction'] == [4]

def test_predict_5():
    X,y = get_processed_data()
    response = app.test_client().post("/predict", json={"image":X[5].tolist()})
    assert response.status_code == 200    
    assert response.get_json()['prediction'] == [9]

def test_predict_6():
    X,y = get_processed_data()
    response = app.test_client().post("/predict", json={"image":X[6].tolist()})
    assert response.status_code == 200    
    assert response.get_json()['prediction'] == [6]

def test_predict_7():
    X,y = get_processed_data()
    response = app.test_client().post("/predict", json={"image":X[7].tolist()})
    assert response.status_code == 200    
    assert response.get_json()['prediction'] == [7]

def test_predict_8():
    X,y = get_processed_data()
    response = app.test_client().post("/predict", json={"image":X[8].tolist()})
    assert response.status_code == 200    
    assert response.get_json()['prediction'] == [8]

def test_predict_9():
    X,y = get_processed_data()
    response = app.test_client().post("/predict", json={"image":X[9].tolist()})
    assert response.status_code == 200    
    assert response.get_json()['prediction'] == [9]