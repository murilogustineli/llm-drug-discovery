# Patient-Trial Matching System: Replication Guide

This guide provides step-by-step instructions to replicate and run the patient-trial matching system.

## Prerequisites

- Python 3.8+ installed
- pip package manager
- Internet connection for downloading datasets and dependencies

## Step 1: Set Up Environment

First, create a directory for the project and set up a virtual environment:

```bash
# Create project directory
mkdir patient_trial_matching
cd patient_trial_matching

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

## Step 2: Install Dependencies

Install all required packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn torch transformers pytorch-lightning
```

## Step 3: Download the Dataset

Download the Heart Failure Clinical Records Dataset:

```bash
# Create data directory
mkdir -p data

# Download dataset
curl -o data/heart_failure_clinical_records_dataset.csv https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv
```

Alternatively, you can download it manually from Kaggle: https://www.kaggle.com/datasets/nimapourmoradi/heart-failure-clinical-records

## Step 4: Project Structure

Create the following directory structure:

```
patient_trial_matching/
├── data/
│   └── heart_failure_clinical_records_dataset.csv
├── plots/
├── venv/
└── scripts/
```

```bash
mkdir -p plots scripts
```

## Step 5: Copy Code Files

Save the following Python scripts to the `scripts` directory:

1. `explore_dataset.py` - For data exploration and preprocessing
2. `simulate_trials.py` - For simulating clinical trials
3. `patient_trial_matching.py` - For the TF-IDF based approach
4. `enhanced_patient_trial_matching.py` - For the LLM-based approach
5. `integrate_structured_data.py` - For structured data integration

You can copy these files from the attachments I provided earlier.

## Step 6: Run the Pipeline

### 6.1. Data Exploration and Preprocessing

```bash
python scripts/explore_dataset.py
```

This script will:
- Load the heart failure dataset
- Perform exploratory data analysis
- Generate summary statistics
- Save preprocessed data to the `data` directory

### 6.2. Simulate Clinical Trials

```bash
python scripts/simulate_trials.py
```

This script will:
- Generate 15 simulated clinical trials with varied eligibility criteria
- Determine eligible patients for each trial
- Split data into training and testing sets
- Save the simulated data to the `data` directory

### 6.3. Run the Simplified Matching Approach

For environments with limited computational resources:

```bash
python scripts/patient_trial_matching.py
```

This script will:
- Process patient and trial data
- Train a TF-IDF vectorizer
- Optimize matching thresholds
- Evaluate matching performance
- Generate visualizations and reports

### 6.4. Run the Advanced LLM-Based Approach

For environments with more computational resources:

```bash
python scripts/enhanced_patient_trial_matching.py
```

Or for the structured data integration version:

```bash
python scripts/integrate_structured_data.py
```

These scripts will:
- Download and use the Bio_ClinicalBERT model
- Process patient and trial data for LLM
- Train the matching model
- Evaluate performance
- Generate visualizations and reports

**Note**: The LLM-based approach requires more computational resources and may take longer to run.

## Step 7: Review Results

After running the scripts, you can find:

- Evaluation metrics in `data/matching_evaluation.csv`
- Matching report in `data/matching_report.txt`
- Visualizations in the `plots` directory

## Troubleshooting

### Common Issues:

1. **Missing dependencies**: Ensure all required packages are installed
   ```bash
   pip install -r requirements.txt  # If you create a requirements file
   ```

2. **CUDA/GPU issues**: If you have GPU issues with PyTorch, try:
   ```bash
   # Force CPU-only mode
   export CUDA_VISIBLE_DEVICES=""
   ```

3. **Memory errors**: For the LLM-based approach, reduce batch size in the code:
   ```python
   # Change batch size from 16 to a smaller number
   train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
   ```

4. **Model download issues**: If you have trouble downloading the Bio_ClinicalBERT model, you can:
   - Check your internet connection
   - Try using a different transformer model by changing the model name in the code

## Customization

To use your own dataset:
1. Replace the dataset file in the `data` directory
2. Modify the data loading and preprocessing code in `explore_dataset.py`
3. Adjust the trial simulation parameters in `simulate_trials.py`

To modify matching parameters:
1. For the simplified approach, adjust thresholds in `simplified_patient_trial_matching.py`
2. For the LLM approach, modify model architecture or training parameters in the respective scripts

## Next Steps

After successfully running the system, consider:
1. Fine-tuning the models with your own data
2. Implementing additional features like explainability
3. Integrating with a web interface for easier use
4. Extending to other clinical domains or datasets
