import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Download necessary NLTK data files
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stopwords set and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.lower() not in stop_words]
    return " ".join(tokens)

# Load paraphraser model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("ramsrigouthamg/t5_paraphraser")
model = AutoModelForSeq2SeqLM.from_pretrained("ramsrigouthamg/t5_paraphraser")
paraphraser = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# Load sentiment analysis models
distil = pipeline("sentiment-analysis")
finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")

def local_paraphrase(text):
    input_text = f"paraphrase: {text} </s>"
    output = paraphraser(input_text, max_length=100, num_return_sequences=1)[0]['generated_text']
    return output

def hybrid_sentiment(text):
    distil_pred = distil(text)[0]
    if distil_pred['score'] < 0.75:
        return finbert(text)[0]
    return distil_pred

def process_headline(headline):
    preprocessed = preprocess(headline)
    paraphrased = local_paraphrase(preprocessed)
    sentiment = hybrid_sentiment(paraphrased)
    return paraphrased, sentiment["label"], sentiment["score"]