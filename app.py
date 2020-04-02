import pickle


from flask import Flask,request,render_template
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import string
def text_process(mess):
    nopunc=[char for char in mess if char not in string.punctuation]
    nopunc="".join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
app=Flask(__name__)
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/results', methods=['POST'])
def results():


    model = pickle.load(open('spam_model.sav', 'rb'))
    bow_transformer = pickle.load(open('spam_vectorizer.sav', 'rb'))



    form = request.form
    if request.method == 'POST':



      #write your function that loads the model
        text = request.form['text']

        y=bow_transformer.transform([text])
        predicted_class = model.predict(y)
        if predicted_class[0]==0:
            return render_template('results.html', text=text,   predicted_cat='spam')
        else :
            return render_template('results.html', text=text,   predicted_cat='ham')

if __name__=='__main__':
     app.run("localhost",9999,debug=True)
