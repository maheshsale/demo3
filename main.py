from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Load Models
text_generator = pipeline(task='text-generation', model='distilgpt2')
summarizer = pipeline(task='summarization', model='facebook/bart-large-cnn')
sentiment_analyzer = pipeline(task='text-classification', model='distilbert-base-uncased-finetuned-sst-2-english')
qa_pipeline = pipeline(task='question-answering') 

class TextRequest(BaseModel):
    text: str

class QARequest(BaseModel):
    context: str
    question: str

@app.get('/')
def home():
    return {
        'message': 'HuggingFace Model Serving API Running'
    }

@app.post('/generate-text')
def generate_text(request: TextRequest):
    result = text_generator(request.text)
    return {
        'generated_text': result[0]['generated_text']
    }

@app.post('/summarize')
def summarize_text(request: TextRequest):
    result = summarizer(request.text)
    return {
        'summarized_text': result[0]['summary_text']
    }

@app.post('/sentiment')
def sentiment(request: TextRequest):
    result = sentiment_analyzer(request.text)
    return {
        'result': result[0]['label']
    }

@app.post('/question-answering')
def question_answering(request: QARequest):
    result = qa_pipeline(question=request.question, context=request.context)
    return {
        'answer': result['answer'],
        'score': result['score']
    }