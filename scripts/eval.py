import torch
import torch.nn as nn
import time
import numpy as np
from sklearn.metrics import f1_score


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    
    all_ner_preds, all_ner_labels = [], []
    all_cls_preds, all_cls_labels = [], []
    
    with torch.no_grad():
        
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs["loss"].item()
            
            ner_logits = outputs["token_logits"].cpu().numpy()
            ner_labels = batch["labels"].cpu().numpy()
            ner_preds = np.argmax(ner_logits, axis=-1) 
            
            mask = ner_labels != -100
            all_ner_preds.extend(ner_preds[mask])
            all_ner_labels.extend(ner_labels[mask])
            
            cls_preds = (outputs["cls_logits"] > 0).int().cpu().numpy()
            all_cls_preds.extend(cls_preds)
            all_cls_labels.extend(batch["cls_labels"].cpu().numpy())
            
    ner_f1 = f1_score(all_ner_labels, all_ner_preds, average="macro")
    
    y_true_cls = np.vstack(all_cls_labels)
    y_pred_cls = np.vstack(all_cls_preds)
    cls_f1 = f1_score(y_true_cls, y_pred_cls, average="micro")
    
    return total_loss / len(dataloader), ner_f1, cls_f1

def run_inference(batch, model, tokenizer, device, id2label, cls_labels_list):
    model.eval()
    
    batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    
    with torch.no_grad():
        start_time = time.time()
        outputs = model(**batch)
        all_time = time.time() - start_time
    
    cls_probs = torch.sigmoid(outputs["cls_logits"])[0].cpu().numpy()
    tokens_preds = torch.argmax(outputs["token_logits"], dim=-1)[0].cpu().numpy()
    input_ids = batch["input_ids"][0].cpu().numpy()
    true_labels = batch["labels"][0].cpu().numpy()
    
    print("CLS probabilities:")
    sorted_cls = sorted(zip(cls_labels_list, cls_probs), key=lambda x: x[1], reverse=True)
    for label_name, prob in sorted_cls[:10]:
        print(f"  {label_name}: {prob:.4f}")
    
    print("\nToken predictions:")
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    current_word = ""
    current_pred_id = -1
    current_true_id = -1
    
    for idx, (token, p_id) in enumerate(zip(tokens, tokens_preds)):  
        t_id = true_labels[idx] if true_labels is not None else -100
        
        if t_id == -100 or token in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
            continue
        
        if token.startswith("##"):
            current_word += token[2:]
        else:
            if current_word:
                pred_label = id2label.get(current_pred_id, "O")
                true_label = id2label.get(current_true_id, "O")
                print(f"  {current_word:<20} : {pred_label:<12}  | True: {true_label}")
            
            current_word = token
            current_pred_id = p_id
            current_true_id = t_id
            
    if current_word:
        pred_label = id2label.get(current_pred_id, "O")
        true_label = id2label.get(current_true_id, "O")
        print(f"  {current_word:<20} : {pred_label:<15}  | True: {true_label}")
            
    return all_time