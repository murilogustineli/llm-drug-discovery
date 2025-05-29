# Enhanced Patient-Trial Matching System

## Project Overview

This project enhances a patient-trial matching system by leveraging large language models (LLMs) and insights from the paper "Large Language Models in Drug Discovery and Development." The system matches patients with appropriate clinical trials based on their medical profiles and trial eligibility criteria, improving the efficiency and accuracy of the matching process.

## Data Source

We used the Heart Failure Clinical Records Dataset from Kaggle, which contains 299 patient records with 13 clinical features including:
- Demographics (age, sex)
- Clinical measurements (ejection fraction, serum creatinine, serum sodium)
- Binary health indicators (anaemia, diabetes, high blood pressure, smoking)

This dataset was selected for its comprehensive clinical features that are ideal for simulating realistic clinical trial eligibility criteria.

## Implementation Approach

### 1. Data Preprocessing and Simulation

- **Patient Data**: Formatted patient records with demographics and medical history
- **Clinical Trials**: Simulated 15 clinical trials with varied eligibility criteria
- **Ground Truth**: Generated eligibility relationships between patients and trials

### 2. Enhanced Matching Approaches

We implemented two approaches for patient-trial matching:

#### A. Advanced LLM-Based Approach
- Uses Bio_ClinicalBERT, a domain-specific language model pre-trained on clinical text
- Enhanced text representation of both patients and trials
- Structured data formatting optimized for LLM processing
- Multi-layer neural network for classification

#### B. Simplified TF-IDF Approach
- Uses TF-IDF vectorization and cosine similarity
- Threshold-based matching
- Optimized for environments with limited computational resources

### 3. Key Enhancements

1. **Domain-Specific Language Understanding**: Leveraged clinical language models that understand medical terminology
2. **Structured Data Representation**: Improved formatting of patient and trial data
3. **Enhanced Feature Engineering**: Better representation of clinical criteria and patient characteristics
4. **Evaluation Framework**: Comprehensive metrics for assessing matching performance

## Results and Evaluation

The evaluation metrics show that matching patients to clinical trials remains challenging:

- **Precision**: How many suggested matches were actually eligible
- **Recall**: How many eligible patients were successfully identified
- **F1 Score**: Harmonic mean of precision and recall

The simplified TF-IDF approach showed limited performance, highlighting the complexity of the matching task and the need for more sophisticated approaches. The advanced LLM-based approach provides a framework for further improvements.

## Files and Deliverables

1. **Data Processing**:
   - `explore_dataset.py`: Initial data exploration and preprocessing
   - `simulate_trials.py`: Clinical trial simulation

2. **Matching Implementations**:
   - `enhanced_patient_trial_matching.py`: Advanced LLM-based approach
   - `integrate_structured_data.py`: Enhanced data structuring for LLMs
   - `simplified_patient_trial_matching.py`: TF-IDF based approach

3. **Outputs**:
   - `/data/matching_evaluation.csv`: Evaluation metrics
   - `/data/matching_report.txt`: Detailed matching report
   - `/plots/`: Visualizations of matching performance

## Future Improvements

1. **Fine-tuning**: Further fine-tune the clinical LLM on trial matching tasks
2. **Multimodal Integration**: Incorporate structured and unstructured data more effectively
3. **Explainability**: Add explanations for why patients match or don't match specific trials
4. **Active Learning**: Implement feedback loops to improve matching over time

## Conclusion

This project demonstrates how LLMs can enhance patient-trial matching by improving the understanding of clinical text and eligibility criteria. While challenges remain, the framework established provides a foundation for more sophisticated matching systems that could significantly improve clinical trial recruitment efficiency.
