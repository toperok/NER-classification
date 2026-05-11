import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, DataCollatorForTokenClassification
import datasets
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import re
import pandas as pd
import seaborn as sns
import phik
from phik import report



def plot_phik_matrix(dataset, classes):

    df_cls = pd.DataFrame(dataset["cls_vec"], columns=classes)

    phik_matrix = df_cls.phik_matrix()

    plt.figure(figsize=(14, 12))
    sns.heatmap(phik_matrix, 
                annot=True, 
                cmap="viridis", 
                annot_kws={"size": 8},
                fmt=".2f",
                linewidths=0.5,
                cbar_kws={'label': 'Phi_K Correlation'})
    
    plt.title("Correlation matrix CLS")
    plt.show()
    plt.close()


def eda(
    dataset: datasets.arrow_dataset.Dataset,
    classes: List[str]
):
    
    classes_sums = np.array(dataset["cls_vec"]).sum(axis=0)
    entity_counts = [len([tag for tag in tags if tag.startswith('B-')]) for tags in dataset['tags']]
    texts_lengths = [len(token) for token in dataset["tokens"]]
    labels_per_doc = np.array(dataset["cls_vec"]).sum(axis=1)
    
    clear_pattern = re.compile(f'\s+')
    for i in range(5):
        clean_text = clear_pattern.sub(' ', dataset[i]['text']).strip()
        
        print(f"Запись {i+1}\n")
        print(f"Текст: {clean_text[:100]}...")
        print(f"Токены: {dataset[i]['tokens'][:7]}...")
        print(f"NER теги: {dataset[i]['tags'][:7]}...")
        print(f"CLS вектор: {dataset[i]['cls_vec']}\n")
        
    print(f"\n{'=' * 70}\n")
    
    for name, class_count in zip(classes, classes_sums):
        print(f"{name}: {class_count}")
        
    print(f"\n{'=' * 70}\n")
    
    print(f"Среднее кол-во меток на документ: {labels_per_doc.mean():.2f}")
    
    print(f"\n{'=' * 70}\n")
    
    plt.figure(figsize=(10, 8))
    plt.barh(classes, classes_sums, color="#7B3DCC", edgecolor="black", linewidth=0.5)
    plt.title("CLS classes")
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()
    plt.close()
    
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    ax1.hist(texts_lengths, bins=30, color="#7B3DCC", edgecolor="black", linewidth=0.5)
    ax1.set_title("Распределение длины текстов")
    ax2.hist(entity_counts, bins=30, color="#7B3DCC", edgecolor="black", linewidth=0.5)
    ax2.set_title("Распределение длины сущностей")
    plt.show()
    plt.close()

    plot_phik_matrix(dataset, classes)
    
    
def create_label_mapping(dataset):
    
    unique_labels = set()
    for example in dataset:
        unique_labels.update(example["tags"])
    unique_labels.add("O")
    
    label_list = sorted(list(unique_labels))
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}
    
    return label2id, id2label
    

def tokenize_and_align_labels(examples, tokenizer, label2id, max_length=512):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=max_length,
        padding=False, # Будем использовать паддинг в DataCollatorForTokenClassification
        is_split_into_words=True,
        return_offsets_mapping=True
    )
    
    labels = []
    for i, tags in enumerate(examples['tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        prev_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != prev_word_idx:
                label_ids.append(label2id[tags[word_idx]])
            else:
                label_ids.append(-100)
            prev_word_idx = word_idx
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    tokenized_inputs["cls_labels"] = examples["cls_vec"]
    tokenized_inputs.pop("offset_mapping") 
    
    return tokenized_inputs


class JointNERDataset(Dataset):
    def __init__(self, tokenized_data):
        self.data = tokenized_data
    
    def __len__(self):
        return len(self.data["input_ids"])
    
    def __getitem__(self, idx):
        return self.data[idx]


class JointCollator(DataCollatorForTokenClassification):
    
    def __call__(self, features):
        cls_labels = [f.pop("cls_labels") for f in features]
        batch = super().__call__(features)
        batch["cls_labels"] = torch.tensor(cls_labels, dtype=torch.float)
        
        return batch


def sanity_check_batch(dataloader, tokenizer, id2label, num_examples=5):
    batch = next(iter(dataloader))
    
    print(f"\n{'=' * 70}\n")
    print("Размерности тензоров")
    print(f"\n{'=' * 70}\n")
    
    print(f"Input ids shape: {batch['input_ids'].shape}")
    print(f"Attention Mask: {batch['attention_mask'].shape}")
    print(f"NER labels shape: {batch['labels'].shape}")
    print(f"CLS labels shape: {batch['cls_labels'].shape}")
    
    print(f"\n{'=' * 70}\n")
    
    for i in range(num_examples):
        print(f"\nПример {i+1}\n")
        print(f"Вектор классификации: {batch['cls_labels'][i][:10].tolist()}")
        
        tokens = tokenizer.convert_ids_to_tokens(batch["input_ids"][i])
        labels = batch["labels"][i].tolist()
        
        print(f"{'Token':<20} | {'Label ID':<10} | {'Tag'}")
        for token, label_id in zip(tokens[:30], labels[:30]):
            if token == tokenizer.pad_token:
                break
            
            tag = id2label[label_id] if label_id != -100 else "IGNORE"
            print(f"{token:<22} {label_id:<12} {tag}")