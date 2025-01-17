import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
import pickle

# Clean data
def get_clean_data():
    data=pd.read_csv('data/data.csv')
    data.drop(["Unnamed: 32", "id"],axis=1, inplace=True,errors='ignore')
    data["diagnosis"]=data["diagnosis"].map({"M":1,"B":0})
    print(data.head(5))
    return data

# Create model
def create_model(data):
    X=data.drop(["diagnosis"],axis=1)
    Y=data["diagnosis"]

    # Scale data
    scaler=StandardScaler()
    x_scaler=scaler.fit_transform(X)

    # split data
    X_train,X_test,Y_train,Y_test = train_test_split(x_scaler,Y,test_size=0.2,random_state=42)

    # Train Model
    model=LogisticRegression()
    model.fit(X_train,Y_train)

    # test model on testing data  
    y_pred=model.predict(X_test)
    accuracy=accuracy_score(Y_test,y_pred)
    class_f=classification_report(Y_test,y_pred)
    print(accuracy)
    print(class_f)
    return model,scaler

# Brain of this file
def main():
    data=get_clean_data()
    model,scaler=create_model(data)
    # Saving Model
    with open('Breast_cancer/model.pkl','wb') as f:
        pickle.dump(model,f)

    with open('Breast_cancer/scaler.pkl','wb') as f:
        pickle.dump(scaler,f)
    

if __name__ == '__main__':
        main()
        
