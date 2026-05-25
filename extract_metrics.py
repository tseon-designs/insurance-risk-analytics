import json
import re

def extract_notebook_outputs():
    outputs = []
    
    # Notebook 1
    try:
        with open('notebooks/01_eda.ipynb', 'r', encoding='utf-8') as f:
            nb1 = json.load(f)
            outputs.append("=== 01 EDA ===")
            for cell in nb1.get('cells', []):
                if cell.get('cell_type') == 'code':
                    for out in cell.get('outputs', []):
                        if 'text' in out:
                            text = "".join(out['text'])
                            if "ACIS PORTFOLIO OVERVIEW" in text or "Loss Ratio by Province" in text or "Total Margin" in text:
                                outputs.append(text)
    except Exception as e:
        outputs.append(f"Error reading 01_eda: {e}")

    # Notebook 2
    try:
        with open('notebooks/02_hypothesis_testing.ipynb', 'r', encoding='utf-8') as f:
            nb2 = json.load(f)
            outputs.append("\n=== 02 HYPOTHESIS TESTING ===")
            for cell in nb2.get('cells', []):
                if cell.get('cell_type') == 'code':
                    for out in cell.get('outputs', []):
                        if 'data' in out and 'text/plain' in out['data']:
                            text = "".join(out['data']['text/plain'])
                            if "Hypothesis" in text or "p-value" in text:
                                outputs.append(text)
                        elif 'text' in out:
                            text = "".join(out['text'])
                            if "HYPOTHESIS TEST RESULTS" in text:
                                outputs.append(text)
    except Exception as e:
        outputs.append(f"Error reading 02_hypothesis_testing: {e}")
        
    # Notebook 3
    try:
        with open('notebooks/03_modeling.ipynb', 'r', encoding='utf-8') as f:
            nb3 = json.load(f)
            outputs.append("\n=== 03 MODELING ===")
            for cell in nb3.get('cells', []):
                if cell.get('cell_type') == 'code':
                    for out in cell.get('outputs', []):
                        if 'text' in out:
                            text = "".join(out['text'])
                            if "Model Comparison" in text or "RMSE" in text or "Risk-Based Premium" in text:
                                outputs.append(text)
                        elif 'data' in out and 'text/plain' in out['data']:
                            text = "".join(out['data']['text/plain'])
                            if "RMSE" in text or "Accuracy" in text or "Feature" in text:
                                outputs.append(text)
    except Exception as e:
        outputs.append(f"Error reading 03_modeling: {e}")

    with open('metrics_summary.txt', 'w', encoding='utf-8') as f:
        f.write("\n".join(outputs))

if __name__ == '__main__':
    extract_notebook_outputs()
