import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split

# Helper function to convert numpy types to Python native types
def convert_to_serializable(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    else:
        return obj

# Load the preprocessed dataset
df = pd.read_csv('/home/ubuntu/data/heart_failure_clinical_records_dataset.csv')

# Define functions to create simulated clinical trials
def generate_trial_protocols(num_trials=10, seed=42):
    """
    Generate simulated clinical trial protocols with eligibility criteria
    based on the heart failure dataset features.
    """
    np.random.seed(seed)
    trials = []
    
    # Define trial types and their focus areas
    trial_types = [
        "Randomized Controlled Trial",
        "Open-Label Study",
        "Double-Blind Study",
        "Observational Study",
        "Phase II Clinical Trial",
        "Phase III Clinical Trial"
    ]
    
    # Define trial names and descriptions
    trial_names = [
        "HEART-PROTECT", "CARDIAC-SHIELD", "FAILURE-PREVENT", 
        "CARDIAC-RECOVERY", "HEART-RESTORE", "CARDIAC-CARE",
        "HEART-FUNCTION", "CARDIAC-BOOST", "FAILURE-REVERSE",
        "HEART-STRENGTH", "CARDIAC-HEALTH", "FAILURE-CONTROL"
    ]
    
    # Define possible interventions
    interventions = [
        "ACE inhibitor therapy", 
        "Beta-blocker treatment",
        "Mineralocorticoid receptor antagonist",
        "SGLT2 inhibitor therapy",
        "Novel anti-inflammatory agent",
        "Stem cell therapy",
        "Gene therapy approach",
        "Cardiac rehabilitation program",
        "Remote monitoring system",
        "Digital health intervention"
    ]
    
    # Define trial durations (in days)
    durations = [90, 180, 365, 730]
    
    # Define criteria ranges based on dataset statistics
    age_ranges = [(40, 65), (50, 75), (65, 95), (40, 95)]
    ef_ranges = [(20, 40), (30, 50), (15, 35), (25, 45)]
    creatinine_ranges = [(0.5, 1.5), (0.7, 2.0), (0.9, 3.0)]
    sodium_ranges = [(130, 145), (125, 140), (135, 150)]
    
    for i in range(num_trials):
        trial_id = f"HF-TRIAL-{i+1:03d}"
        trial_name = np.random.choice(trial_names) + f"-{i+1}"
        trial_type = np.random.choice(trial_types)
        intervention = np.random.choice(interventions)
        duration = int(np.random.choice(durations))  # Convert to Python int
        
        # Randomly select criteria combinations using Python's random instead of numpy for tuples
        import random
        age_range = random.choice(age_ranges)
        ef_range = random.choice(ef_ranges)
        creatinine_range = random.choice(creatinine_ranges)
        sodium_range = random.choice(sodium_ranges)
        
        # Randomly decide which binary criteria to include
        include_anaemia = bool(np.random.choice([True, False]))
        include_diabetes = bool(np.random.choice([True, False]))
        include_hbp = bool(np.random.choice([True, False]))
        include_sex = bool(np.random.choice([True, False]))
        include_smoking = bool(np.random.choice([True, False]))
        
        # For included binary criteria, randomly decide the required value
        anaemia_value = int(np.random.choice([0, 1])) if include_anaemia else None
        diabetes_value = int(np.random.choice([0, 1])) if include_diabetes else None
        hbp_value = int(np.random.choice([0, 1])) if include_hbp else None
        sex_value = int(np.random.choice([0, 1])) if include_sex else None
        smoking_value = int(np.random.choice([0, 1])) if include_smoking else None
        
        # Create eligibility criteria text
        eligibility_criteria = []
        eligibility_criteria.append(f"Age between {age_range[0]} and {age_range[1]} years")
        eligibility_criteria.append(f"Ejection fraction between {ef_range[0]}% and {ef_range[1]}%")
        eligibility_criteria.append(f"Serum creatinine between {creatinine_range[0]} and {creatinine_range[1]} mg/dL")
        eligibility_criteria.append(f"Serum sodium between {sodium_range[0]} and {sodium_range[1]} mEq/L")
        
        if include_anaemia:
            eligibility_criteria.append(f"{'With' if anaemia_value == 1 else 'Without'} anaemia")
        if include_diabetes:
            eligibility_criteria.append(f"{'With' if diabetes_value == 1 else 'Without'} diabetes")
        if include_hbp:
            eligibility_criteria.append(f"{'With' if hbp_value == 1 else 'Without'} high blood pressure")
        if include_sex:
            eligibility_criteria.append(f"{'Male' if sex_value == 1 else 'Female'} patients only")
        if include_smoking:
            eligibility_criteria.append(f"{'Current smokers' if smoking_value == 1 else 'Non-smokers'} only")
        
        # Create trial description
        description = f"A {trial_type.lower()} investigating the efficacy of {intervention} "
        description += f"in heart failure patients. The study will last for {duration} days."
        
        # Create trial object
        trial = {
            "id": trial_id,
            "name": trial_name,
            "type": trial_type,
            "description": description,
            "intervention": intervention,
            "duration": duration,
            "eligibility_criteria": {
                "text": ". ".join(eligibility_criteria),
                "age_min": int(age_range[0]),
                "age_max": int(age_range[1]),
                "ef_min": int(ef_range[0]),
                "ef_max": int(ef_range[1]),
                "creatinine_min": float(creatinine_range[0]),
                "creatinine_max": float(creatinine_range[1]),
                "sodium_min": int(sodium_range[0]),
                "sodium_max": int(sodium_range[1]),
                "anaemia": anaemia_value,
                "diabetes": diabetes_value,
                "high_blood_pressure": hbp_value,
                "sex": sex_value,
                "smoking": smoking_value
            },
            "eligible_patients": []  # Will be filled later
        }
        
        trials.append(trial)
    
    return trials

def determine_eligible_patients(trials, patient_data):
    """
    Determine which patients are eligible for each trial based on the eligibility criteria.
    """
    for trial in trials:
        criteria = trial["eligibility_criteria"]
        eligible_patients = []
        
        for _, patient in patient_data.iterrows():
            # Check continuous criteria
            if not (criteria["age_min"] <= patient["age"] <= criteria["age_max"]):
                continue
            if not (criteria["ef_min"] <= patient["ejection_fraction"] <= criteria["ef_max"]):
                continue
            if not (criteria["creatinine_min"] <= patient["serum_creatinine"] <= criteria["creatinine_max"]):
                continue
            if not (criteria["sodium_min"] <= patient["serum_sodium"] <= criteria["sodium_max"]):
                continue
            
            # Check binary criteria if specified
            if criteria["anaemia"] is not None and patient["anaemia"] != criteria["anaemia"]:
                continue
            if criteria["diabetes"] is not None and patient["diabetes"] != criteria["diabetes"]:
                continue
            if criteria["high_blood_pressure"] is not None and patient["high_blood_pressure"] != criteria["high_blood_pressure"]:
                continue
            if criteria["sex"] is not None and patient["sex"] != criteria["sex"]:
                continue
            if criteria["smoking"] is not None and patient["smoking"] != criteria["smoking"]:
                continue
            
            # If all criteria are met, add patient to eligible list - convert to int to avoid numpy int64
            eligible_patients.append(int(patient.name))
        
        trial["eligible_patients"] = [int(pid) for pid in eligible_patients]  # Convert all to Python int
    
    return trials

def format_patient_data(df):
    """
    Format patient data for use in the matching algorithm.
    """
    patients = []
    
    for idx, row in df.iterrows():
        patient = {
            "id": int(idx),
            "demographics": f"{'Male' if row['sex'] == 1 else 'Female'}, {int(row['age'])} years old",
            "medical_history": ""
        }
        
        # Build medical history text
        conditions = []
        if row["anaemia"] == 1:
            conditions.append("Anaemia")
        if row["diabetes"] == 1:
            conditions.append("Diabetes")
        if row["high_blood_pressure"] == 1:
            conditions.append("High Blood Pressure")
        if row["smoking"] == 1:
            conditions.append("Smoker")
        
        # Add clinical measurements
        measurements = [
            f"Ejection Fraction: {int(row['ejection_fraction'])}%",
            f"Serum Creatinine: {float(row['serum_creatinine'])} mg/dL",
            f"Serum Sodium: {int(row['serum_sodium'])} mEq/L",
            f"Creatinine Phosphokinase: {int(row['creatinine_phosphokinase'])} mcg/L",
            f"Platelets: {int(row['platelets'])} cells/mL"
        ]
        
        # Combine conditions and measurements
        if conditions:
            patient["medical_history"] += ", ".join(conditions)
        if conditions and measurements:
            patient["medical_history"] += ". "
        patient["medical_history"] += ". ".join(measurements)
        
        patients.append(patient)
    
    return patients

# Generate trial protocols
trials = generate_trial_protocols(num_trials=15)

# Determine eligible patients for each trial
trials = determine_eligible_patients(trials, df)

# Format patient data
patients = format_patient_data(df)

# Split data into training and testing sets
train_patients, test_patients = train_test_split(patients, test_size=0.2, random_state=42)
train_trials, test_trials = train_test_split(trials, test_size=0.2, random_state=42)

# Convert all data to JSON serializable format
train_patients = convert_to_serializable(train_patients)
test_patients = convert_to_serializable(test_patients)
train_trials = convert_to_serializable(train_trials)
test_trials = convert_to_serializable(test_trials)

# Save the simulated data
with open('/home/ubuntu/data/train_patients.json', 'w') as f:
    json.dump(train_patients, f, indent=2)

with open('/home/ubuntu/data/test_patients.json', 'w') as f:
    json.dump(test_patients, f, indent=2)

with open('/home/ubuntu/data/train_trials.json', 'w') as f:
    json.dump(train_trials, f, indent=2)

with open('/home/ubuntu/data/test_trials.json', 'w') as f:
    json.dump(test_trials, f, indent=2)

# Print summary statistics
print(f"Generated {len(trials)} clinical trial protocols")
print(f"Formatted data for {len(patients)} patients")
print(f"Training set: {len(train_patients)} patients, {len(train_trials)} trials")
print(f"Testing set: {len(test_patients)} patients, {len(test_trials)} trials")

# Print eligibility statistics
eligibility_counts = [len(trial["eligible_patients"]) for trial in trials]
print(f"Average eligible patients per trial: {np.mean(eligibility_counts):.2f}")
print(f"Min eligible patients: {np.min(eligibility_counts)}")
print(f"Max eligible patients: {np.max(eligibility_counts)}")

# Print example trial
print("\nExample Trial Protocol:")
example_trial = trials[0]
print(f"Trial ID: {example_trial['id']}")
print(f"Trial Name: {example_trial['name']}")
print(f"Type: {example_trial['type']}")
print(f"Description: {example_trial['description']}")
print(f"Eligibility Criteria: {example_trial['eligibility_criteria']['text']}")
print(f"Number of Eligible Patients: {len(example_trial['eligible_patients'])}")

# Print example patient
print("\nExample Patient:")
example_patient = patients[0]
print(f"Patient ID: {example_patient['id']}")
print(f"Demographics: {example_patient['demographics']}")
print(f"Medical History: {example_patient['medical_history']}")
