from flask import Flask, request
from transformers import BertTokenizerFast, BertForSequenceClassification
import re
import string
app = Flask(__name__)
app.config["DEBUG"] = True

model_name = "bert-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=True)

model1 = BertForSequenceClassification.from_pretrained("results/checkpoint-375", num_labels=20)

target_names = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
                'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles',
                'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space',
                'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc',
                'talk.religion.misc']
def text_cleaner(text):
    '''some text cleaning method'''

    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)

    return text


@app.route("/predict", methods=["POST"])
def get_prediction():
    text = request.json
    print(text)
    text = text[0]["text"]
    text=text_cleaner(text)
    # prepare our text into tokenized sequence
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    # perform inference to our model
    outputs = model1(**inputs)
    # get output probabilities by doing softmax
    probs = outputs[0].softmax(1)
    # executing argmax function to get the candidate label
    return target_names[probs.argmax()]


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
