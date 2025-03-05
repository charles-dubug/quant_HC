import utils
import json
import os
import datetime

file_path = r"f:\AI\quant_HC\news_summary_adata\capital_flow_restructured.json"

with open(file_path, 'r', encoding='utf-8') as file:
    capital_flow_data = json.load(file)

output_file = r"f:\AI\quant_HC\news_summary_adata\predictions.json"

predictions = []
if os.path.exists(output_file):
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
        print(f"Loaded {len(predictions)} existing predictions.")
    except json.JSONDecodeError:
        print("Error loading existing predictions file. Starting fresh.")

for data in capital_flow_data:
    if any(existing['index_code'] == data['index_code'] for existing in predictions):
        print(f"Skipping {data['index_name']} (code: {data['index_code']}) - already processed")
        continue
    
    prompt = utils.get_data_analysis_prompt(data)
    print('Analyzing...')
    prediction = utils.analysis_with_deepseek_robust(prompt)
    prediction['timestamp'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    predictions.append(prediction)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"Processed {prediction['index_name']}")

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(predictions, f, ensure_ascii=False, indent=2)