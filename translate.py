from googletrans import Translator
import ast
import json
translator = Translator()  
# ENGLISH_DATA = "../class_descrs/newsgroups/combined_ng_manual_train.labels";
english_data_path = "./test_data/test_train.labels"

# source venv/bin/activate
# python3 translate.py > ./test_data/test_translated_train.labels

with open(english_data_path) as f:
    data = f.read()
    chunks = data.split('\n')
with open("sample.json", "w") as outfile:
    for c in chunks:
        dict = json.loads(c)
        translated_text = translator.translate(dict['text'], src='en',dest='es')
        translated = {"text": translated_text.text, "label": dict['label']}
        #json.dump(translated, outfile, indent=0)
        print(translated)
    