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