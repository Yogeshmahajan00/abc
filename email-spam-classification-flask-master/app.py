from flask import Flask,render_template,request
import pickle

cv = pickle.load(open("model/vectorizer.pkl","rb"))
clf = pickle.load(open("model/model.pkl","rb"))


app = Flask(__name__)

@app.route('/')
def index():
    # Vectorize the input
    #result = cv.transform([sample]).toarray()
    # Predict
    #pred = clf.predict(result)
    #print(pred)
    return render_template("index.html")

@app.route('/predict',methods=['post'])
def predict():
    
    userInput = request.form.get('email')
    result = cv.transform([userInput]).toarray()
    # Predict
    pred = clf.predict(result)
    pred = int(pred[0])
    if pred == 0:
        pred=-1
    return render_template("index.html",label=pred)

if __name__ == "__main__":
    app.run(debug=True)



# from flask import Flask,render_template,request
# app=Flask(__name__)
# import pickle
# cv=pickle.load(open("model/vectorizer.pkl","rb"))
# clf=pickle.load(open("model/model.pkl","rb"))
# @app.route("/")
# def index():
#     return render_template("index.html")
# @app.route("/predict",method=["POST"])

# def result():
#     user_input=request.form.get("email")
#     trans=cv.transform([user_input]).toarray()
#     pred=clf.predict(trans)
#     pred=int(pred[0])
#     if pred == 0:
#         pred=-1
#     return render_template("index.html",label=pred)

# if __name__=="__main__":
#     app.run(debug=True)
