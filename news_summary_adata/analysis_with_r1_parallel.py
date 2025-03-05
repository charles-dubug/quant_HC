import utils
import json
import os
import datetime
import concurrent.futures
import time
import signal

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

# Filter out data that has already been processed
data_to_process = []
for data in capital_flow_data:
    if any(existing['index_code'] == data['index_code'] for existing in predictions):
        print(f"Skipping {data['index_name']} (code: {data['index_code']}) - already processed")
    else:
        data_to_process.append(data)

print(f"Processing {len(data_to_process)} items in parallel...")

def process_data(data):
    try:
        prompt = utils.get_data_analysis_prompt(data)
        # print(f"Analyzing {data['index_name']} (code: {data['index_code']})...")
        prediction = utils.analysis_with_deepseek_robust(prompt)
        prediction['timestamp'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return prediction
    except Exception as e:
        print(f"Error processing {data['index_name']}: {str(e)}")
        return {
            "index_code": data.get('index_code', ''),
            "index_name": data.get('index_name', ''),
            "prediction": "",
            "reason": f"处理失败: {str(e)}",
            "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

# Number of parallel workers - adjust based on your system capabilities and API rate limits
max_workers = 3

# Process data in parallel
try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_data = {executor.submit(process_data, data): data for data in data_to_process}
        
        for future in concurrent.futures.as_completed(future_to_data):
            data = future_to_data[future]
            try:
                prediction = future.result()
                if prediction:
                    predictions.append(prediction)
                    # print(f"Processed {prediction['index_name']}")
                    
                    # Save after each successful prediction to avoid data loss
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(predictions, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"Exception occurred while processing {data['index_name']}: {str(e)}")
except KeyboardInterrupt:
    print("\nGracefully shutting down... Saving current progress...")
    # Final save of all predictions
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print("Progress saved. Program stopped.")
    exit(0)

# Final save of all predictions
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(predictions, f, ensure_ascii=False, indent=2)

print("All processing complete!")