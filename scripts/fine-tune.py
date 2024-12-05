import json
from datasets import Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments


# Load and preprocess data
def load_data(file_path):
   with open(file_path, 'r', encoding='utf-8') as f:
       data = [json.loads(line) for line in f]
   return Dataset.from_list(data)


# Tokenize the dataset
def tokenize_function(examples):
   questions = [q.strip() for q in examples["question"]]
   inputs = tokenizer(
       ["question: " + q + " context: " + c for q, c in zip(questions, examples["context"])],
       max_length=512,
       truncation=True,
       padding="max_length",
       return_tensors="pt",
   )
  
   with tokenizer.as_target_tokenizer():
       labels = tokenizer(
           examples["answer"],
           max_length=64,
           truncation=True,
           padding="max_length",
           return_tensors="pt",
       )


   inputs["labels"] = labels["input_ids"]
   return inputs


# Main fine-tuning function
def fine_tune():
   global tokenizer
   # Load data
   full_dataset = load_data("../data/processed_data.json")


   # Split the dataset into training and evaluation sets
   # train_val_dataset = full_dataset.train_test_split(test_size=0.2, random_state=42)
   train_val_dataset = full_dataset.train_test_split(test_size=0.2, seed=42)
   train_dataset = train_val_dataset['train']
   eval_dataset = train_val_dataset['test']


   # Initialize tokenizer and model
   model_name = "google/flan-t5-small"
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


   # Tokenize datasets
   tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
   tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=eval_dataset.column_names)


   # Define training arguments
   training_args = Seq2SeqTrainingArguments(
       output_dir="./models/fine_tuned_llm",
       evaluation_strategy="epoch",
       learning_rate=5e-5,
       per_device_train_batch_size=8,
       per_device_eval_batch_size=8,
       num_train_epochs=3,
       weight_decay=0.01,
       predict_with_generate=True,
       push_to_hub=False,
       logging_dir='./logs',
       logging_steps=100,
       save_total_limit=2,
   )


   # Initialize Trainer
   trainer = Seq2SeqTrainer(
       model=model,
       args=training_args,
       train_dataset=tokenized_train_dataset,
       eval_dataset=tokenized_eval_dataset,
       tokenizer=tokenizer,
   )


   # Fine-tune the model
   trainer.train()


   # Save the fine-tuned model
   trainer.save_model("./models/fine_tuned_llm")
   tokenizer.save_pretrained("./models/fine_tuned_llm")


if __name__ == "__main__":
   fine_tune()

