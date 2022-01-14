import streamlit as st
import joblib

st.title('Sentiment Analysis G8 DigitalHouse')
st.markdown('This app analyses the sentiment behind a sentence.')

classifier = joblib.load('sentiment-model.pkl')

def run():
    with st.form(key='Type sentence'):
        sentence = st.text_input('Type a sentence for which you want to know the sentiment')
        submit_button = st.form_submit_button(label='Submit')
        
        if submit_button:
            result = classifier.predict((['i love mandarinas']))
            st.write(result)
 

if __name__=='__main__':
    run()