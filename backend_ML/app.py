import os
from flask import Flask, request, jsonify
from transformers import pipeline
import whisper
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

# модель Арины
pipe = None
# whisper
to_text_model = None
# детектор токсичности
toxic_tokenizer = None
toxic_model = None
# детектор эмоций
emotions_tokenizer = None
emotions_model = None

best_thresholds = [0.36734693877551017, 0.2857142857142857, 0.2857142857142857, 0.16326530612244897, 0.14285714285714285, 0.14285714285714285, 0.18367346938775508, 0.3469387755102041, 0.32653061224489793, 0.22448979591836732, 0.2040816326530612, 0.2857142857142857, 0.18367346938775508, 0.2857142857142857, 0.24489795918367346, 0.7142857142857142, 0.02040816326530612, 0.3061224489795918, 0.44897959183673464, 0.061224489795918366, 0.18367346938775508, 0.04081632653061224, 0.08163265306122448, 0.1020408163265306, 0.22448979591836732, 0.3877551020408163, 0.3469387755102041, 0.24489795918367346]
LABELS = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']
ID2LABEL = dict(enumerate(LABELS))

def load_models():
    global pipe, to_text_model, toxic_tokenizer, toxic_model, emotions_tokenizer, emotions_model
    pipe = pipeline("audio-classification", model="ericguan04/distilhubert-finetuned-ravdess")
    to_text_model = whisper.load_model("base")
    # токсичность
    PATH = 'khvatov/ru_toxicity_detector'
    toxic_tokenizer = AutoTokenizer.from_pretrained(PATH)
    toxic_model = AutoModelForSequenceClassification.from_pretrained(PATH)

    if torch.cuda.is_available():
        toxic_model.to(torch.device('cuda'))
    else:
        toxic_model.to(torch.device("cpu"))
    # эмоции
    emotions_tokenizer = AutoTokenizer.from_pretrained("fyaronskiy/ruRoberta-large-ru-go-emotions")
    emotions_model = AutoModelForSequenceClassification.from_pretrained("fyaronskiy/ruRoberta-large-ru-go-emotions")

    if torch.cuda.is_available():
        emotions_model.to(torch.device('cuda'))
    else:
        emotions_model.to(torch.device("cpu"))

def split_text_into_sentences(text):
    sentences = re.split(r'(?<=\.|\!|\?)\s+', text)
    return sentences

def get_toxicity_probs(text):
    with torch.no_grad():
        inputs = toxic_tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(toxic_model.device)
        proba = torch.nn.functional.softmax(toxic_model(**inputs).logits, dim=1).cpu().numpy()
    return proba[0]

def predict_emotions(text):
    inputs = emotions_tokenizer(text, truncation=True, add_special_tokens=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
        logits = emotions_model(**inputs).logits
    probas = torch.sigmoid(logits).squeeze(dim=0)
    class_binary_labels = (probas > torch.tensor(best_thresholds)).int()
    return [ID2LABEL[label_id] for label_id, value in enumerate(class_binary_labels) if value == 1]

def analyze_wav(file_path):
    try:
        # модель первая
        speech = pipe(file_path)
        speech_output = [item['label'] for item in speech if item['score'] > 0.3]
        # переводим в предложения
        text = to_text_model.transcribe(file_path)["text"]
        sentences = split_text_into_sentences(text)

        toxicity_result = []
        emotions_result = []

        for s in sentences:
            # узнаем токсичность
            toxic_proba = get_toxicity_probs(s)[0]
            if toxic_proba > 0.5:
                toxicity_result.append('toxic')
            else:
                toxicity_result.append('not_toxic')

            # распознаем эмоции
            emotions_result.append(predict_emotions(s))

        return {
            'speech_model': speech_output,
            'toxicity_model': toxicity_result,
            'emotions_model': emotions_result
        }

    except Exception as e:
        return {'error': str(e)}

@app.route('/analyze', methods=['POST'])
def analyze_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # вставить свой путь
    file_path = os.path.join('uploads', file.filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file.save(file_path)

    analysis_result = analyze_wav(file_path)

    os.remove(file_path)

    return jsonify(analysis_result)

if __name__ == '__main__':
    load_models()
    app.run(debug=True)
