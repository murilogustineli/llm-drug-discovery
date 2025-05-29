import json
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load patient and trial data
patient_data = json.load(open("patient_data.json"))
trial_data = json.load(open("trial_data.json"))

# Split data into training and testing sets
train_patient_data, test_patient_data, train_trial_data, test_trial_data = (
    train_test_split(patient_data, trial_data, test_size=0.2, random_state=42)
)

# Define the LLM model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")


# Define a custom dataset class for patient-trial matching
class PatientTrialDataset(Dataset):
    def __init__(self, patient_data, trial_data, tokenizer):
        self.patient_data = patient_data
        self.trial_data = trial_data
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        patient = self.patient_data[idx]
        trial = self.trial_data[idx]
        patient_text = patient["demographics"] + " " + patient["medical_history"]
        trial_text = trial["eligibility_criteria"] + " " + trial["trial_design"]
        inputs = self.tokenizer.encode_plus(
            patient_text,
            trial_text,
            add_special_tokens=True,
            max_length=512,
            return_attention_mask=True,
            return_tensors="pt",
        )
        labels = torch.tensor(1 if patient["id"] in trial["eligible_patients"] else 0)
        return {
            "input_ids": inputs["input_ids"].flatten(),
            "attention_mask": inputs["attention_mask"].flatten(),
            "labels": labels,
        }

    def __len__(self):
        return len(self.patient_data)


# Define a PyTorch Lightning module for patient-trial matching
class PatientTrialModule(pl.LightningModule):
    def __init__(self):
        super(PatientTrialModule, self).__init__()
        self.bert = model
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        outputs = self.classifier(pooled_output)
        return outputs

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        outputs = self(input_ids, attention_mask)
        loss = torch.nn.CrossEntropyLoss()(outputs, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        outputs = self(input_ids, attention_mask)
        loss = torch.nn.CrossEntropyLoss()(outputs, labels)
        self.log("val_loss", loss)
        _, predicted = torch.max(outputs, dim=1)
        accuracy = accuracy_score(labels.cpu().numpy(), predicted.cpu().numpy())
        self.log("val_accuracy", accuracy)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5)


# Create datasets and data loaders for training and testing
train_dataset = PatientTrialDataset(train_patient_data, train_trial_data, tokenizer)
test_dataset = PatientTrialDataset(test_patient_data, test_trial_data, tokenizer)
train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize the PyTorch Lightning module and trainer
module = PatientTrialModule()
trainer = pl.Trainer(
    max_epochs=5, callbacks=[EarlyStopping(monitor="val_loss", patience=3)]
)

# Train the model
trainer.fit(module, train_data_loader)

# Evaluate the model
trainer.test(module, test_data_loader)


# Use the model to match patients with clinical trials
def match_patients_with_trials(patient_data, trial_data):
    patient_trial_dataset = PatientTrialDataset(patient_data, trial_data, tokenizer)
    patient_trial_data_loader = DataLoader(
        patient_trial_dataset, batch_size=32, shuffle=False
    )
    module.eval()
    matched_trials = []
    with torch.no_grad():
        for batch in patient_trial_data_loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            outputs = module(input_ids, attention_mask)
            _, predicted = torch.max(outputs, dim=1)
            for i, prediction in enumerate(predicted):
                if prediction == 1:
                    matched_trials.append(
                        (batch["input_ids"][i], batch["attention_mask"][i])
                    )
    return matched_trials


# Example dataset
patient_data = [
    {
        "id": 1,
        "demographics": "Male, 35 years old",
        "medical_history": "Diabetes, Hypertension",
    },
    {
        "id": 2,
        "demographics": "Female, 28 years old",
        "medical_history": "Asthma, Allergies",
    },
    {
        "id": 3,
        "demographics": "Male, 42 years old",
        "medical_history": "Heart Disease, High Cholesterol",
    },
]

trial_data = [
    {
        "id": 1,
        "eligibility_criteria": "Diabetes, Hypertension",
        "trial_design": "Randomized controlled trial",
    },
    {
        "id": 2,
        "eligibility_criteria": "Asthma, Allergies",
        "trial_design": "Open-label study",
    },
    {
        "id": 3,
        "eligibility_criteria": "Heart Disease, High Cholesterol",
        "trial_design": "Double-blind study",
    },
]

matched_trials = match_patients_with_trials(patient_data, trial_data)
print(matched_trials)

# And on a lighter note, why did the AI program go to therapy? Because it was feeling a little "glitchy"!
