from enum import Enum
from typing import Dict, Optional, Tuple, Union, Any, List

import torch
import torch.nn as nn
from pydantic import BaseModel, Field
from transformers import RobertaConfig, RobertaModel, RobertaForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput


class RerankerType(str, Enum):
    """Enum for reranker types."""
    POINTWISE = "pointwise"
    PAIRWISE = "pairwise"


class RerankerConfig(BaseModel):
    """Configuration for CodeBERT Reranker."""
    model_name_or_path: str = Field(..., description="Path to pretrained model or model name")
    config: Optional[RobertaConfig] = Field(None, description="Model configuration")
    margin: float = Field(0.3, description="Margin for pairwise ranking loss")
    reranker_type: RerankerType = Field(RerankerType.POINTWISE, description="Reranking approach")

    model_config = {"arbitrary_types_allowed": True}

class CodeBERTReranker(nn.Module):
    """
    CodeBERT-based reranker for code search
    
    This model takes a query and code candidate as input and outputs a relevance score.
    For fine-tuning, it can be trained with pointwise or pairwise ranking approaches.
    """
    
    def __init__(self, 
                 model_name_or_path: str = 'microsoft/codebert-base', 
                 config: Optional[RobertaConfig] = None, 
                 margin: float = 0.3, 
                 reranker_type: Union[str, RerankerType] = RerankerType.POINTWISE):
        """
        Initialize the reranker model
        
        Args:
            model_name_or_path: Path to pretrained model or model name (e.g., 'microsoft/codebert-base')
            config: Model configuration (RobertaConfig)
            margin: Margin for pairwise ranking loss
            reranker_type: 'pointwise' or 'pairwise' reranking approach
        """
        super(CodeBERTReranker, self).__init__()
        
        # Validate and convert string to enum if needed
        if isinstance(reranker_type, str):
            reranker_type = RerankerType(reranker_type)
        
        self.reranker_type = reranker_type
        self.margin = margin
        
        if config is None:
            self.config = RobertaConfig.from_pretrained(model_name_or_path)
        else:
            self.config = config
            
        # For pointwise reranking, we use the standard sequence classification model
        if reranker_type == RerankerType.POINTWISE:
            self.model = RobertaForSequenceClassification.from_pretrained(
                model_name_or_path, 
                config=self.config
            )
        # For pairwise reranking, we use the base model and add our own classification head
        else:
            self.model = RobertaModel.from_pretrained(model_name_or_path)
            self.classifier = nn.Linear(self.config.hidden_size, 1)
            
    def forward(self, 
                input_ids: Optional[torch.Tensor] = None, 
                attention_mask: Optional[torch.Tensor] = None, 
                token_type_ids: Optional[torch.Tensor] = None, 
                position_ids: Optional[torch.Tensor] = None, 
                labels: Optional[torch.Tensor] = None, 
                pos_input_ids: Optional[torch.Tensor] = None, 
                pos_attention_mask: Optional[torch.Tensor] = None, 
                pos_token_type_ids: Optional[torch.Tensor] = None,
                neg_input_ids: Optional[torch.Tensor] = None, 
                neg_attention_mask: Optional[torch.Tensor] = None, 
                neg_token_type_ids: Optional[torch.Tensor] = None
               ) -> Union[SequenceClassifierOutput, Dict[str, torch.Tensor]]:
        """
        Forward pass
        
        For pointwise reranking:
            - Input: query-code pair
            - Output: relevance score
        
        For pairwise reranking:
            - Input: query with positive code example and negative code example
            - Output: margin ranking loss
        """
        
        if self.reranker_type == RerankerType.POINTWISE:
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
    
    def get_score(self, 
                  input_ids: torch.Tensor,
                  attention_mask: torch.Tensor,
                  token_type_ids: Optional[torch.Tensor] = None
                 ) -> torch.Tensor:
        """
        Get relevance score for a query-code pair
        Used for inference in both pointwise and pairwise reranking
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs (optional)
            
        Returns:
            Relevance scores
        """
        if self.reranker_type == RerankerType.POINTWISE:
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