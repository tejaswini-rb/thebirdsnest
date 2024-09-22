import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, TrainerCallback
import numpy as np
import evaluate
from EarlyBird import EarlyBird
from undecayed import undecayed_pruning
from gradient import gradient_based_pruning

import time

import torch.nn.utils.prune as prune

to_prune = False

class UndecayedTrainer(Trainer):
    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        self.accelerator.backward(loss)

        if (self.state.global_step == self.state.max_steps - 1) and (to_prune == True):
            for _, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    undecayed_pruning(module, epsilon=0.01, amount=0.7)

        return loss.detach() / self.args.gradient_accumulation_steps

class GradientTrainer(Trainer):
    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        self.accelerator.backward(loss)

        if (self.state.global_step == self.state.max_steps - 1) and (to_prune == True):
            for _, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    gradient_based_pruning(module, amount=0.7)

        return loss.detach() / self.args.gradient_accumulation_steps


metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-uncased", num_labels=2).cuda()

dataset = load_dataset("nyu-mll/glue", "sst2")



tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

train_dataset = dataset["train"]
test_dataset = dataset["validation"]

dataset_size = len(train_dataset) // 5

# don't train over the entire dataset
train_dataset = train_dataset.select(range(dataset_size))

def tokenize(examples):
    return tokenizer(examples["sentence"], truncation=True)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

training_args = TrainingArguments(output_dir="test_trainer", num_train_epochs=1, evaluation_strategy='epoch')


# gradient - prune rate 0.3, 0.5, 0.7, 0.9       
# undecayed - prune rate 0.3, 0.5, 0.7, 0.9



#we run early bird one epoch at a time, once we find early bird, we quit and go to pruning
# trainer = Trainer(
# trainer = GradientTrainer(
# trainer = UndecayedTrainer(
earlyBird = EarlyBird(0.7)

start_time = time.time()

for epochs in range(20):
    if earlyBird.early_bird_emerge(model):
        print("Early Bird Found!")

        to_prune = True

        trainer = UndecayedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        trainer.train()


        # Output epoch number
        print(f"Epoch: {epochs}")
        break

    trainer = UndecayedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
print("This is Magnitude Pruning, with prune rate 0.7")

results = trainer.evaluate(eval_dataset=test_dataset)
print("Pre-Pruning Accuracy: " + str(results['eval_accuracy']))
print("Pre-Pruning Loss: " + str(results['eval_loss']))
print("Pre-Pruning Runtime: " + str(results['eval_runtime']))

#magnitude - prune rate 0.3, 0.5, 0.7, 0.9 (uncomment for magnitude pruning)
# for _, module in model.named_modules():
#     if isinstance(module, nn.Linear):
#         prune.L1Unstructured.apply(module, name="weight", amount=0.7)
#         prune.L1Unstructured.apply(module, name="bias", amount=0.7)

    
# Retraining
results = trainer.evaluate(eval_dataset=test_dataset)

print("Post-Pruning + Pre-Training Accuracy: " + str(results['eval_accuracy']))
print("Post-Pruning + Pre-Training Loss: " + str(results['eval_loss']))
print("Post-Pruning + Pre-Training Runtime: " + str(results['eval_runtime']))
to_prune = False

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

end_time = time.time()

results = trainer.evaluate(eval_dataset=test_dataset)

print("Post-Prune + Retrain Accuracy: " + str(results['eval_accuracy']))
print("Post-Prune + Retrain Loss: " + str(results['eval_loss']))
print("Post-Prune + Retrain Runtime: " + str(results['eval_runtime']))

print("Time (in seconds): " + str(end_time - start_time))