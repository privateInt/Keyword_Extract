import json
import torch
from transformers import PreTrainedTokenizerFast
from transformers.models.bart import BartForConditionalGeneration
from flask import Flask, jsonify, request

app = Flask(__name__)

model = BartForConditionalGeneration.from_pretrained('./kobart_summary')
tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1')

def infer(text):
    input_ids = tokenizer.encode(text)
    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.unsqueeze(0)
    output = model.generate(input_ids, eos_token_id=1, max_length=512, num_beams=5)
    output = tokenizer.decode(output[0], skip_special_tokens=True)
    return output

@app.route('/keyword_extract', methods=['POST'])
def keyword_extract():
    data_dict = json.loads(request.data)
    
    prompt = data_dict['prompt']
    result = infer(prompt)
    
    result_dict = {
        'prompt': prompt,
        'result': result
    }
    
    return json.dumps(result_dict, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=8459)
