import chunk
from googletrans import Translator
import json
import sys
import time
translator = Translator()  
# ENGLISH_DATA = "../class_descrs/newsgroups/combined_ng_manual_train.labels";
# english_data_path = "./test_data/test_train.labels"
english_data_path = sys.argv[1]

# source venv/bin/activate
# python3 translate.py > ./test_data/test_translated_train.labels

with open(english_data_path) as f:
    data = f.read()
    chunks = data.split('\n')

for json_str in chunks:
# for i,c in enumerate(chunks):
#     if(i%4 !=0):
#         continue
#     json_str = chunks[i]+chunks[i+1]+chunks[i+2]+chunks[i+3]
    
    dict = json.loads(json_str)
    translated_text = translator.translate(dict['text'], src='en',dest='es')
    # translated_text = translator.translate(dict['text'], src='en',dest='zh-cn')
    translated_text.text = translated_text.text.replace('"',"'")
    translated = {"text": translated_text.text, "label": dict['label']}
    json_string = json.dumps(translated, ensure_ascii=False).encode('utf8')
    print(json_string.decode())
    # print(json.dumps(translated, ensure_ascii=False).encode('utf8'))
    time.sleep(1)
    