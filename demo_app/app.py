import pickle
import string
import numpy as np
import nltk
import sklearn
import gradio as gr
from nltk.corpus import stopwords


# File Paths
model_path = 'rf_model.sav' 
bow_vectorizer_path = "vectorizer"
tfidf_path = "tfidf_transformer"

# Loading the files
model = pickle.load(open(model_path, 'rb'))
bow_vectorizer = pickle.load(open(bow_vectorizer_path, 'rb'))
tfidf_transformer = pickle.load(open(tfidf_path, 'rb'))
nltk.download("stopwords")

labels = ["not spam", "spam"]

# declerating the example case
Examples = [
    "Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...",
    "Dear John, thank you for submitting your application for the open position. We appreciate your interest and will be in touch soon regarding next steps",
    "Hi Sarah, just wanted to follow up on our meeting last week and see if you had any further questions or concerns. Let me know if there's anything else I can help you with.",
    "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's",
    "Congratulations! You have been selected to receive a free trip to the Bahamas. To claim your prize, simply click on the link below and fill out the registration form.",
    "URGENT: Your account has been compromised. Please click on the link below to reset your password and secure your account.",
]

# Util functions
def text_preprocessing(text):

    # getting the stopwords
    STOPWORDS = stopwords.words('english') + ['u', 'ü', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure']
    
    # selecting non puction words
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    # clearning and joining the words to create text
    clean_text = ' '.join([word for word in nopunc.split() if word.lower() not in STOPWORDS])
    
    return clean_text

def vector_transformation(sentence):
  return bow_vectorizer.transform(sentence)
def tfidf_transformation(bow):
  return tfidf_transformer.transform(bow)

def predict(text):

  # preparing the input into convenient form
  sentence = text_preprocessing(text)
  
  # converting the text into numerical representation
  bow = bow_vectorizer.transform([sentence])
  features = tfidf_transformation(bow)

  # prediction
  probabilities = model.predict_proba(features) #.predict(features)
  probs = probabilities.flatten()

  # output form
  results = {l : np.round(p, 3) for l, p in zip(labels, probs)}

  return results

# GUI Component
demo = gr.Interface(predict, "text", "label", examples = Examples)

# Launching the demo
if __name__ == "__main__":
    demo.launch()
