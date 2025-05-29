import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create plots directory if it doesn't exist
os.makedirs('/home/ubuntu/plots', exist_ok=True)

# Load the simulated patient and trial data
with open('/home/ubuntu/data/train_patients.json', 'r') as f:
    train_patient_data = json.load(f)

with open('/home/ubuntu/data/train_trials.json', 'r') as f:
    train_trial_data = json.load(f)

with open('/home/ubuntu/data/test_patients.json', 'r') as f:
    test_patient_data = json.load(f)

with open('/home/ubuntu/data/test_trials.json', 'r') as f:
    test_trial_data = json.load(f)

# Enhanced data formatting for LLM processing
def format_patient_for_llm(patient):
    """Format patient data in a structured way optimized for LLM processing"""
    # Extract demographic information
    demographics = patient['demographics'].split(', ')
    gender = demographics[0]
    age = demographics[1]
    
    # Parse medical history into structured components
    medical_history = patient['medical_history']
    conditions = []
    measurements = []
    
    # Split by periods and process each component
    components = medical_history.split('. ')
    for component in components:
        if component:
            if any(keyword in component for keyword in ['Fraction', 'Creatinine', 'Sodium', 'Phosphokinase', 'Platelets']):
                measurements.append(component)
            elif component:
                conditions.append(component)
    
    # Create structured format
    structured_patient = {
        "id": patient['id'],
        "gender": gender,
        "age": age,
        "conditions": conditions,
        "measurements": measurements
    }
    
    # Generate enhanced text representation
    enhanced_text = f"Patient ID: {patient['id']}\n"
    enhanced_text += f"Demographics: {gender}, {age}\n"
    
    if conditions:
        enhanced_text += "Medical Conditions:\n"
        for condition in conditions:
            enhanced_text += f"- {condition}\n"
    
    if measurements:
        enhanced_text += "Clinical Measurements:\n"
        for measurement in measurements:
            enhanced_text += f"- {measurement}\n"
    
    return enhanced_text, structured_patient

def format_trial_for_llm(trial):
    """Format trial data in a structured way optimized for LLM processing"""
    # Extract eligibility criteria into structured components
    criteria_text = trial['eligibility_criteria']['text']
    criteria_components = criteria_text.split('. ')
    
    # Categorize criteria
    demographic_criteria = []
    clinical_criteria = []
    condition_criteria = []
    
    for component in criteria_components:
        if component:
            if 'Age' in component:
                demographic_criteria.append(component)
            elif any(keyword in component for keyword in ['Male', 'Female']):
                demographic_criteria.append(component)
            elif any(keyword in component for keyword in ['Ejection', 'Creatinine', 'Sodium']):
                clinical_criteria.append(component)
            else:
                condition_criteria.append(component)
    
    # Create structured format
    structured_trial = {
        "id": trial['id'],
        "name": trial['name'],
        "type": trial['type'],
        "description": trial['description'],
        "demographic_criteria": demographic_criteria,
        "clinical_criteria": clinical_criteria,
        "condition_criteria": condition_criteria
    }
    
    # Generate enhanced text representation
    enhanced_text = f"Trial ID: {trial['id']}\n"
    enhanced_text += f"Name: {trial['name']}\n"
    enhanced_text += f"Type: {trial['type']}\n"
    enhanced_text += f"Description: {trial['description']}\n"
    
    enhanced_text += "Eligibility Criteria:\n"
    
    if demographic_criteria:
        enhanced_text += "Demographic Requirements:\n"
        for criterion in demographic_criteria:
            enhanced_text += f"- {criterion}\n"
    
    if clinical_criteria:
        enhanced_text += "Clinical Requirements:\n"
        for criterion in clinical_criteria:
            enhanced_text += f"- {criterion}\n"
    
    if condition_criteria:
        enhanced_text += "Condition Requirements:\n"
        for criterion in condition_criteria:
            enhanced_text += f"- {criterion}\n"
    
    return enhanced_text, structured_trial

# Process all patients and trials
print("Processing patient data for LLM...")
processed_train_patients = []
for patient in train_patient_data:
    enhanced_text, structured_patient = format_patient_for_llm(patient)
    processed_train_patients.append({
        "id": patient['id'],
        "original": patient,
        "structured": structured_patient,
        "enhanced_text": enhanced_text
    })

processed_test_patients = []
for patient in test_patient_data:
    enhanced_text, structured_patient = format_patient_for_llm(patient)
    processed_test_patients.append({
        "id": patient['id'],
        "original": patient,
        "structured": structured_patient,
        "enhanced_text": enhanced_text
    })

print("Processing trial data for LLM...")
processed_train_trials = []
for trial in train_trial_data:
    enhanced_text, structured_trial = format_trial_for_llm(trial)
    processed_train_trials.append({
        "id": trial['id'],
        "original": trial,
        "structured": structured_trial,
        "enhanced_text": enhanced_text,
        "eligible_patients": trial['eligible_patients']
    })

processed_test_trials = []
for trial in test_trial_data:
    enhanced_text, structured_trial = format_trial_for_llm(trial)
    processed_test_trials.append({
        "id": trial['id'],
        "original": trial,
        "structured": structured_trial,
        "enhanced_text": enhanced_text,
        "eligible_patients": trial['eligible_patients']
    })

# Save processed data
with open('/home/ubuntu/data/processed_train_patients.json', 'w') as f:
    json.dump(processed_train_patients, f, indent=2)

with open('/home/ubuntu/data/processed_test_patients.json', 'w') as f:
    json.dump(processed_test_patients, f, indent=2)

with open('/home/ubuntu/data/processed_train_trials.json', 'w') as f:
    json.dump(processed_train_trials, f, indent=2)

with open('/home/ubuntu/data/processed_test_trials.json', 'w') as f:
    json.dump(processed_test_trials, f, indent=2)

print("Data processing complete. Enhanced data saved to /home/ubuntu/data/")

# Define the tokenizer and model
# Using ClinicalBERT instead of standard BERT for better medical domain understanding
print("Loading ClinicalBERT model...")
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
base_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

# Enhanced dataset class for patient-trial matching with structured data
class StructuredPatientTrialDataset(Dataset):
    def __init__(self, processed_patients, processed_trials, tokenizer, max_length=512):
        self.processed_patients = processed_patients
        self.processed_trials = processed_trials
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pairs = self._create_pairs()
        
    def _create_pairs(self):
        """Create all possible patient-trial pairs with labels"""
        pairs = []
        for trial in self.processed_trials:
            eligible_patients = set(trial["eligible_patients"])
            
            # Add positive examples (eligible patients)
            for patient in self.processed_patients:
                if patient["id"] in eligible_patients:
                    pairs.append((patient, trial, 1))
            
            # Add negative examples (non-eligible patients)
            # Balance the dataset by sampling a similar number of negative examples
            non_eligible_patients = [p for p in self.processed_patients if p["id"] not in eligible_patients]
            num_positive = len([p for p in self.processed_patients if p["id"] in eligible_patients])
            # Sample at most the same number of negative examples as positive ones
            num_negative = min(num_positive, len(non_eligible_patients))
            if num_negative > 0:
                sampled_negative = np.random.choice(non_eligible_patients, num_negative, replace=False)
                for patient in sampled_negative:
                    pairs.append((patient, trial, 0))
        
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        patient, trial, label = self.pairs[idx]
        
        # Use enhanced text representations
        patient_text = patient["enhanced_text"]
        trial_text = trial["enhanced_text"]
        
        # Tokenize with special tokens and attention masks
        encoding = self.tokenizer.encode_plus(
            patient_text,
            trial_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
            "patient_id": patient["id"],
            "trial_id": trial["id"]
        }

# Enhanced PyTorch Lightning module for patient-trial matching
class StructuredPatientTrialModule(pl.LightningModule):
    def __init__(self, base_model, learning_rate=2e-5):
        super(StructuredPatientTrialModule, self).__init__()
        self.save_hyperparameters()
        self.bert = base_model
        self.dropout = nn.Dropout(0.1)
        
        # Enhanced architecture with additional layers
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2)
        )
        
        # Track metrics
        self.train_acc = []
        self.val_acc = []
        self.train_loss = []
        self.val_loss = []
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits
    
    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        logits = self(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(logits, labels)
        
        # Calculate accuracy
        _, predicted = torch.max(logits, dim=1)
        acc = (predicted == labels).float().mean()
        
        # Log metrics
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        
        self.train_loss.append(loss.item())
        self.train_acc.append(acc.item())
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        logits = self(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(logits, labels)
        
        # Calculate accuracy
        _, predicted = torch.max(logits, dim=1)
        acc = (predicted == labels).float().mean()
        
        # Log metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        
        self.val_loss.append(loss.item())
        self.val_acc.append(acc.item())
        
        return {"val_loss": loss, "val_acc": acc, "predictions": predicted, "labels": labels}
    
    def on_validation_epoch_end(self):
        # Store predictions and labels as instance attributes
        self.all_preds = []
        self.all_labels = []
        
        # Collect predictions and labels from validation_step outputs
        outputs = self.validation_step_outputs
        for output in outputs:
            if "val_loss" in output:
                self.val_loss_value = output["val_loss"]
            if "predictions" in output:
                self.all_preds.append(output["predictions"])
            if "labels" in output:
                self.all_labels.append(output["labels"])
        
        # Concatenate all predictions and labels
        if self.all_preds and self.all_labels:
            self.all_preds = torch.cat(self.all_preds)
            self.all_labels = torch.cat(self.all_labels)
            
            # Calculate precision, recall, f1
            precision, recall, f1, _ = precision_recall_fscore_support(
                self.all_labels.cpu().numpy(), 
                self.all_preds.cpu().numpy(), 
                average='binary'
            )
            
            # Log aggregated metrics
            self.log("val_precision", precision)
            self.log("val_recall", recall)
            self.log("val_f1", f1)
    
    def configure_optimizers(self):
        # Use AdamW optimizer with weight decay and learning rate scheduler
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }
    
    def plot_metrics(self):
        """Plot training and validation metrics"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_loss, label='Train Loss')
        plt.plot(self.val_loss, label='Val Loss')
        plt.title('Loss over epochs')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_acc, label='Train Accuracy')
        plt.plot(self.val_acc, label='Val Accuracy')
        plt.title('Accuracy over epochs')
        plt.xlabel('Batch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('/home/ubuntu/plots/training_metrics.png')
        plt.close()

# Function to match patients with clinical trials
def match_patients_with_trials(model, processed_patients, processed_trials, tokenizer, threshold=0.5):
    """
    Match patients with clinical trials using the trained model
    Returns a dictionary of trial_id -> list of matched patient_ids
    """
    model.eval()
    matches = {}
    
    for trial in processed_trials:
        trial_id = trial["id"]
        matches[trial_id] = []
        
        for patient in processed_patients:
            patient_id = patient["id"]
            
            # Use enhanced text representations
            patient_text = patient["enhanced_text"]
            trial_text = trial["enhanced_text"]
            
            # Tokenize
            encoding = tokenizer.encode_plus(
                patient_text,
                trial_text,
                add_special_tokens=True,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt"
            )
            
            # Get prediction
            with torch.no_grad():
                input_ids = encoding["input_ids"]
                attention_mask = encoding["attention_mask"]
                outputs = model(input_ids, attention_mask)
                probabilities = torch.softmax(outputs, dim=1)
                match_probability = probabilities[0][1].item()  # Probability of class 1 (match)
                
                # If probability exceeds threshold, consider it a match
                if match_probability >= threshold:
                    matches[trial_id].append({
                        "patient_id": patient_id,
                        "match_probability": match_probability
                    })
    
    return matches

# Function to evaluate matching performance
def evaluate_matching(matches, processed_trials):
    """
    Evaluate the performance of the matching algorithm
    """
    results = {
        "trial_id": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "num_predicted": [],
        "num_actual": [],
        "num_correct": []
    }
    
    for trial in processed_trials:
        trial_id = trial["id"]
        actual_matches = set(trial["eligible_patients"])
        predicted_matches = set([m["patient_id"] for m in matches.get(trial_id, [])])
        
        # Calculate metrics
        correct_matches = actual_matches.intersection(predicted_matches)
        
        precision = len(correct_matches) / len(predicted_matches) if predicted_matches else 0
        recall = len(correct_matches) / len(actual_matches) if actual_matches else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Store results
        results["trial_id"].append(trial_id)
        results["precision"].append(precision)
        results["recall"].append(recall)
        results["f1"].append(f1)
        results["num_predicted"].append(len(predicted_matches))
        results["num_actual"].append(len(actual_matches))
        results["num_correct"].append(len(correct_matches))
    
    # Calculate overall metrics
    overall_precision = np.mean(results["precision"])
    overall_recall = np.mean(results["recall"])
    overall_f1 = np.mean(results["f1"])
    
    print(f"Overall Precision: {overall_precision:.4f}")
    print(f"Overall Recall: {overall_recall:.4f}")
    print(f"Overall F1 Score: {overall_f1:.4f}")
    
    return pd.DataFrame(results)

# Create datasets with processed data
print("Creating datasets...")
train_dataset = StructuredPatientTrialDataset(processed_train_patients, processed_train_trials, tokenizer)
val_dataset = StructuredPatientTrialDataset(processed_test_patients, processed_test_trials, tokenizer)

print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Initialize model
model = StructuredPatientTrialModule(base_model)

# Initialize trainer with early stopping and checkpointing
checkpoint_callback = ModelCheckpoint(
    dirpath='/home/ubuntu/data/checkpoints',
    filename='patient-trial-matching-{epoch:02d}-{val_f1:.2f}',
    save_top_k=1,
    monitor='val_f1',
    mode='max'
)

trainer = pl.Trainer(
    max_epochs=5,
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=2),
        checkpoint_callback
    ],
    enable_progress_bar=True,
    enable_model_summary=True
)

# Train the model
print("Starting model training...")
trainer.fit(model, train_loader, val_loader)
print("Training complete!")

# Plot training metrics
model.plot_metrics()

# Evaluate on test set
print("Evaluating model on test set...")
matches = match_patients_with_trials(model, processed_test_patients, processed_test_trials, tokenizer)
evaluation_df = evaluate_matching(matches, processed_test_trials)

# Save evaluation results
evaluation_df.to_csv('/home/ubuntu/data/matching_evaluation.csv', index=False)

# Visualize evaluation results
plt.figure(figsize=(10, 6))
sns.barplot(x='trial_id', y='f1', data=evaluation_df)
plt.title('F1 Score by Trial')
plt.xlabel('Trial ID')
plt.ylabel('F1 Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/home/ubuntu/plots/f1_by_trial.png')

# Plot precision-recall comparison
plt.figure(figsize=(10, 6))
width = 0.35
x = np.arange(len(evaluation_df))
plt.bar(x - width/2, evaluation_df['precision'], width, label='Precision')
plt.bar(x + width/2, evaluation_df['recall'], width, label='Recall')
plt.xlabel('Trial ID')
plt.ylabel('Score')
plt.title('Precision and Recall by Trial')
plt.xticks(x, evaluation_df['trial_id'], rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('/home/ubuntu/plots/precision_recall_by_trial.png')

# Save the trained model
torch.save(model.state_dict(), '/home/ubuntu/data/patient_trial_matching_model.pt')

print("Evaluation complete! Results saved to /home/ubuntu/data/matching_evaluation.csv")
print("Visualizations saved to /home/ubuntu/plots/")
