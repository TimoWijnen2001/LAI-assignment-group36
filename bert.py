import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, set_seed
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from torch import nn
from WeightedLossTrainer import WeightedLossTrainer

# 'name': prefix of the filename (e.g., 'birth' -> 'birth_train.csv')
# 'target': the specific column name in that csv to predict
tasks = [
    {'name': 'birth',       'target': 'birth_year'},
    {'name': 'extrovert',   'target': 'extrovert'},
    {'name': 'feeling',     'target': 'feeling'},
    {'name': 'gender',      'target': 'female'},  # Note: Column is 'female', not 'gender'
    {'name': 'judging',     'target': 'judging'},
    {'name': 'nationality', 'target': 'nationality'},
    {'name': 'political',   'target': 'political_leaning'},
    {'name': 'sensing',     'target': 'sensing'}
]

# The two input columns we want to compare
input_types = ['post', 'post_masked']

# Set seed for reproducibility
random_state = 1234
torch.manual_seed(random_state)
torch.cuda.manual_seed_all(random_state)
set_seed(random_state)

# Enable cuda
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize tokenizer once
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

def tokenize_func(batch, input_col_name):
    # Dynamic tokenization based on which column we are currently testing
    return tokenizer(batch[input_col_name], padding="max_length", truncation=True, max_length=512)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # Average='macro' handles both binary and multi-class (like nationality) well
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# --- Main Training Loop ---
for task in tasks:
    preds_by_input = {}
    metrics_by_input = {}

    task_name = task['name']
    target_col = task['target']
    
    print(f"\n{'='*60}")
    print(f"Processing Task: {task_name} (Target: {target_col})")
    print(f"{'='*60}")

    try:
        df_train = pd.read_csv(f'cleaned_data/train/{task_name}_train.csv')
        df_test = pd.read_csv(f'cleaned_data/test/{task_name}_test.csv')
        df_val = pd.read_csv(f'cleaned_data/val/{task_name}_val.csv')
    except FileNotFoundError as e:
        print(f"Skipping {task_name}: File not found ({e})")
        continue

    # --- Label Encoding ---
    # Define valid labels based on Training Data
    unique_labels = sorted(df_train[target_col].unique())
    label2id = {label: i for i, label in enumerate(unique_labels)}
    num_labels = len(unique_labels)
    
    # Map and Clean Training Data
    df_train['label'] = df_train[target_col].map(label2id)
    df_train = df_train.dropna(subset=['label'])
    df_train['label'] = df_train['label'].astype(int)  # <--- CRITICAL FIX: Force int type

    # Map and Clean Test/Val Data
    df_test['label'] = df_test[target_col].map(label2id)
    df_val['label'] = df_val[target_col].map(label2id)
    
    df_test = df_test.dropna(subset=['label'])
    df_val = df_val.dropna(subset=['label'])
    
    df_test['label'] = df_test['label'].astype(int) # Force int type
    df_val['label'] = df_val['label'].astype(int)   # Force int type

    # Calculate Weights
    counts = df_train['label'].value_counts().sort_index()
    weights = len(df_train) / (len(counts) * counts.values)
    class_weights = torch.tensor(weights, dtype=torch.float)

    for input_col in input_types:
        print(f"\n--- Training on column: {input_col} ---")
        
        # Datasets
        train_ds = Dataset.from_pandas(df_train[[input_col, 'label']].rename(columns={input_col: 'text'}), preserve_index=False)
        val_ds = Dataset.from_pandas(df_val[[input_col, 'label']].rename(columns={input_col: 'text'}), preserve_index=False)
        test_ds = Dataset.from_pandas(df_test[[input_col, 'label']].rename(columns={input_col: 'text'}), preserve_index=False)

        # Tokenization
        tokenized_train = train_ds.map(lambda x: tokenize_func(x, 'text'), batched=True, load_from_cache_file=False)
        tokenized_val = val_ds.map(lambda x: tokenize_func(x, 'text'), batched=True, load_from_cache_file=False)
        tokenized_test = test_ds.map(lambda x: tokenize_func(x, 'text'), batched=True, load_from_cache_file=False)

        # Model
        model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=num_labels)
        
        output_dir = f"./results/{task_name}_{input_col}"

        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            fp16=torch.cuda.is_available(),
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=1, 
            seed=random_state, 
            data_seed =random_state, 
        )

        trainer = WeightedLossTrainer(
            class_weights=class_weights,
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        
        # Eval
        test_results = trainer.predict(tokenized_test)
        metrics = compute_metrics(test_results)
        preds = test_results.predictions.argmax(-1)
        

        preds_by_input[input_col] = preds
        metrics_by_input[input_col] = metrics

        if "post" in preds_by_input and "post_masked" in preds_by_input:
            same = (preds_by_input["post"] == preds_by_input["post_masked"]).mean()
            print(f"\nPrediction equality (post vs post_masked): {same:.4f}")


        print(f"Result for {task_name} ({input_col}): Accuracy={metrics['accuracy']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")
    