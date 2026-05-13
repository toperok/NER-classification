class JointModel(nn.Module):
    
    def __init__(
        self,
        model_name,
        label2id,
        id2label,
        num_cls_labels=30,
        use_uncertainty_weight=True
    ) -> None:
        super().__init__()
        
        self.label2id = label2id
        self.id2label = id2label
        self.use_uncertainty_weight = use_uncertainty_weight
        
        self.model = AutoModel.from_pretrained(model_name)
        hidden_size = self.model.config.hidden_size
        
        self.dropout = nn.Dropout(0.1)
        self.token_head = nn.Linear(hidden_size, len(self.label2id))
        
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_cls_labels)
        )
        
        self.feature_norm = nn.LayerNorm(hidden_size * 2)
        
        self.log_sigma_token = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_cls = nn.Parameter(torch.tensor(0.0))
        
        nn.init.normal_(self.token_head.weight, std=0.02)
        nn.init.zeros_(self.token_head.bias)
        for l in self.cls_head:
            if isinstance(l, nn.Linear):
                nn.init.normal_(l.weight, std=0.02)
                nn.init.zeros_(l.bias)
    
    
    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None,
        cls_labels=None,
    ):
        outputs = self.model(input_ids, attention_mask)
        last_hidden_state = outputs.last_hidden_state
        
        token_output = self.dropout(last_hidden_state)
        token_logits = self.token_head(token_output)
        
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()

        sum_embeddings = torch.sum(last_hidden_state * mask, 1)
        mean_pooling = sum_embeddings / torch.clamp(mask.sum(1), min=1e-9)

        masked_state = last_hidden_state * mask - (1 - mask) * 1e10
        max_pooling, _ = torch.max(masked_state, dim=1)
        
        combined_features = torch.cat([mean_pooling, max_pooling], dim=1)
        combined_features = self.feature_norm(combined_features)
        
        cls_logits = self.cls_head(self.dropout(combined_features))
        
        loss = None
        if labels is not None and cls_labels is not None:
            
            token_loss = F.cross_entropy(
                token_logits.view(-1, len(self.label2id)),
                labels.view(-1),
                ignore_index=-100
            )
            cls_loss = F.binary_cross_entropy_with_logits(
                cls_logits,
                cls_labels
            )
            
            if self.use_uncertainty_weight:
                loss = (torch.exp(-2.0 * self.log_sigma_token) * token_loss + self.log_sigma_token) + \
                       (torch.exp(-2.0 * self.log_sigma_cls) * cls_loss + self.log_sigma_cls) * 2
            else:
                loss = 0.5 * token_loss + 1.0 * cls_loss 

        return {
            "loss": loss, 
            "token_logits": token_logits, 
            "cls_logits": cls_logits
        }