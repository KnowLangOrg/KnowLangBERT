import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel, RobertaForSequenceClassification

class CodeBERTReranker(nn.Module):
    """
    CodeBERT-based reranker for code search
    
    This model takes a query and code candidate as input and outputs a relevance score.
    For fine-tuning, it can be trained with pointwise or pairwise ranking approaches.
    """
    
    def __init__(self, model_name_or_path, config=None, margin=0.3, reranker_type="pariwise"):
        """
        Initialize the reranker model
        
        Args:
            model_name_or_path: Path to pretrained model or model name (e.g., 'microsoft/codebert-base')
            config: Model configuration (RobertaConfig)
            margin: Margin for pairwise ranking loss
            reranker_type: 'pointwise' or 'pairwise' reranking approach
        """
        super(CodeBERTReranker, self).__init__()
        
        self.reranker_type = reranker_type
        self.margin = margin
        
        if config is None:
            self.config = RobertaConfig.from_pretrained(model_name_or_path)
        else:
            self.config = config
            
        # For pointwise reranking, we use the standard sequence classification model
        if reranker_type == "pointwise":
            self.model = RobertaForSequenceClassification.from_pretrained(
                model_name_or_path, 
                config=self.config
            )
        # For pairwise reranking, we use the base model and add our own classification head
        else:
            self.model = RobertaModel.from_pretrained(model_name_or_path)
            self.classifier = nn.Linear(self.config.hidden_size, 1)
            
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, 
                position_ids=None, labels=None, 
                pos_input_ids=None, pos_attention_mask=None, pos_token_type_ids=None,
                neg_input_ids=None, neg_attention_mask=None, neg_token_type_ids=None):
        """
        Forward pass
        
        For pointwise reranking:
            - Input: query-code pair
            - Output: relevance score
        
        For pairwise reranking:
            - Input: query with positive code example and negative code example
            - Output: margin ranking loss
        """
        
        if self.reranker_type == "pointwise":
            # Standard sequence classification
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                labels=labels
            )
            
            return outputs  # (loss), logits, (hidden_states), (attentions)
            
        else:  # pairwise reranking
            # Process positive example
            pos_outputs = self.model(
                input_ids=pos_input_ids,
                attention_mask=pos_attention_mask,
                token_type_ids=pos_token_type_ids
            )
            
            # Process negative example
            neg_outputs = self.model(
                input_ids=neg_input_ids,
                attention_mask=neg_attention_mask,
                token_type_ids=neg_token_type_ids
            )
            
            # Get CLS token representation for both examples
            pos_pooled = pos_outputs.last_hidden_state[:, 0]
            neg_pooled = neg_outputs.last_hidden_state[:, 0]
            
            # Calculate scores
            pos_score = self.classifier(pos_pooled)
            neg_score = self.classifier(neg_pooled)
            
            # Calculate loss
            loss = None
            if labels is not None:
                # Margin ranking loss
                loss_fn = nn.MarginRankingLoss(margin=self.margin)
                # All labels should be 1 as positive should be ranked higher
                target = torch.ones_like(pos_score)
                loss = loss_fn(pos_score, neg_score, target)
            
            return {
                'loss': loss,
                'pos_score': pos_score,
                'neg_score': neg_score
            }
    
    def get_score(self, input_ids=None, attention_mask=None, token_type_ids=None):
        """
        Get relevance score for a query-code pair
        Used for inference in both pointwise and pairwise reranking
        """
        if self.reranker_type == "pointwise":
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            logits = outputs.logits
            # For binary classification, return the positive class score
            if logits.shape[-1] == 2:
                return logits[:, 1]
            return logits
            
        else:  # pairwise reranking
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            pooled = outputs.last_hidden_state[:, 0]
            score = self.classifier(pooled)
            return score