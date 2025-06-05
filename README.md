# Patient–Trial Matching Notebooks

This repository contains Jupyter notebooks for simulating clinical trials and matching simulated patients to those trials using both **TF-IDF** and [**Bio_ClinicalBERT**](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) embeddings. All work is organized under the `notebooks/` folder.

This project was powered by **LobeChat** using a **custom OpenAI-compatible API key** connected to a **Meta-Llama-3.3-7B-Instruct** model.  
> The model is hosted on **Denvr Dataworks**, leveraging **Intel's Gaudi 2 accelerator** for efficient inference.

![lobe-chat](./images/lobe-chat.png)

To learn more about enterprise-scale inference, available models, and how to deploy your own AI workloads with Gaudi, visit [https://www.denvrdata.com/intel](https://www.denvrdata.com/intel)


## Repo structure

```
llm-drug-discovery/
├── data/
├── notebooks/
├── plots/
├── requirements.txt
└── README.md
```


- **`data/`**: 
  - Raw CSV from UCI (heart_failure_clinical_records_dataset.csv) and generated JSON files (`train_patients.json`, `train_trials.json`, etc.).
  - Evaluation results (`matching_eval.csv`).

- **`notebooks/`**:
  1. **`00_explore_dataset.ipynb`** – load and explore the original heart failure dataset.  
  2. **`01_simulate_trials.ipynb`** – generate synthetic clinical trial protocols and match patients to those trials using deterministic rules.  
  3. **`02_patient_trial_matching.ipynb`** – baseline TF-IDF text‐similarity approach to match patients and trials, with threshold tuning and per‐trial F1 evaluation.  
  4. **`03_enhanced_patient_trial_matching.ipynb`** – enhanced matching using Bio_ClinicalBERT embeddings combined with a rule‐based prefilter; threshold tuning and per‐trial F1 evaluation.

- **`plots/`**:
  - Pre‐generated figures (F1 vs. threshold, precision/recall) used in the notebooks.

- **`requirements.txt`**: 
  - List of Python packages required to run all notebooks (e.g. `pandas`, `numpy`, `scikit-learn`, `torch`, `transformers`, etc.).

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/murilogustineli/llm-drug-discovery.git
cd llm-drug-discovery
```

### 2. Set Up a Virtual Environment

It’s recommended to create and activate a Python 3.8+ virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate      # on macOS/Linux
# .venv\Scripts\activate       # on Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

All required libraries (e.g., `pandas`, `matplotlib`, `scikit-learn`, `torch`, `transformers`) are listed in `requirements.txt`.

### 4. Run the Notebooks

Navigate to the `notebooks/` folder, then run each notebook in order:
1. `00_explore_datasetipynb`
2. `01_simulate_trials.ipynb`
3. `02_patient_trial_matching.ipynb`
4. `03_enhanced_patient_trial_matching.ipynb`

Each notebook is self-contained and walks you through:
- Loading and preprocessing the data
- Creating simulated clinical trial protocols
- Performing patient–trial matching with TF-IDF or ClinicalBERT
- Evaluating matching performance (precision, recall, F1)
- Visualizing results (threshold curves, per‐trial F1 bar charts)

## Results & Plots

- `plots/f1_by_trial.png` and `plots/precision_recall_by_trial.png` show TF-IDF–based matching outcomes.
- `plots/threshold_vs_f1.png` plots F1 score versus threshold for TF-IDF.
- `plots/bert_threshold_vs_f1.png` and `plots/bert_f1_by_trial.png` show equivalent results when using ClinicalBERT embeddings.

You can regenerate these figures by rerunning the corresponding notebook cells.

## Customization

    To change the number of simulated trials or patient samples, modify parameters in 01_simulate_trials.ipynb.

    To adjust TF-IDF or ClinicalBERT threshold ranges, edit the threshold arrays in 02_patient_trial_matching.ipynb and 03_enhanced_patient_trial_matching.ipynb.

    Feel free to add new notebooks or scripts for alternative matching methods or other dataset explorations.

## Paper & Context

This work was inspired by methods in clinical trial matching, but focuses solely on the “Clinical Trials” stage. The original heart failure dataset comes from [**UCI: Heart Failure Clinical Records Data Set**](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records).
