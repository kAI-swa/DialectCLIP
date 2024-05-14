import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput
from typing import Optional, List, Union
from .configuration_dialectclip import DialectCLIPConfig
from .utils_dialectclip import dialect_clip_loss, casuallm_loss


def gumbel_softmax(
        logits: torch.Tensor, 
        tau: float = 1, 
        dim: int = -2
):
    '''
    -----------------
    Inputs:
        logits: attention score shape of [query_seq_length, key_seq_length]
        tau: temperature hyperparameters
        dim: dimension to compute softmax and argmax
    '''
    gumbels = torch.tensor(np.random.gumbel(0, 1, size=logits.shape), dtype=logits.dtype, device=logits.device)
    gumbels = (logits + gumbels) / tau
    y_soft = F.softmax(gumbels, dim=dim)

    index = torch.argmax(y_soft, dim=dim, keepdim=True)
    y_hard = torch.scatter(torch.zeros_like(logits, dtype=logits.dtype), dim=dim, index=index, src=torch.ones_like(index, dtype=logits.dtype))
    attn = y_hard + y_soft - y_soft.detach()
    return attn


class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            gumbel_softmax: Optional[bool] = None,
            attn_dropout_rate: float = 0.2,
            mlp_dropout_rate: float = 0.2
    ):
        super().__init__()
        self.gumbel_softmax = gumbel_softmax
        self.attn_dropout_rate = attn_dropout_rate
        self.mlp_dropout_rate = mlp_dropout_rate

        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(p=self.attn_dropout_rate)
        self.mlp_drop = nn.Dropout(p=self.mlp_dropout_rate)

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            gumbel_tau: float = 1.0,
            dim: int = -2,
            return_attn: Optional[bool] = None,
            *args, **kwargs
    ):
        '''
        ----------------------
        Inputs:
            gumbel_tau: temperature hyperparameters
        '''
        batch_size, num_group_tokens, hidden_size = query.shape
        scale_factor = hidden_size ** -0.5
        raw_attn = torch.matmul(query, key.swapaxes(-1, -2)) * scale_factor
        if self.gumbel_softmax:
            attn = gumbel_softmax(raw_attn, tau=gumbel_tau, dim=dim)
        else:
            attn = F.softmax(raw_attn, dim=dim)
        
        attn = self.attn_drop(attn) # shape: [batch_size, num_group_tokens, speech_seq_length]
        assert attn.shape == (batch_size, num_group_tokens, key.shape[1]), f"shape incorrect, attn shape {attn.shape}"
        output = torch.matmul(attn, value)
        output = self.mlp_drop(self.proj(output))
        if return_attn:
            output_attn = attn
        else:
            output_attn = None
        return output, output_attn
    

class GroupingLayer(nn.Module):
    def __init__(
            self,
            speech_hidden_size,
            text_hidden_size,
            num_group_tokens: int = 64,
            attn_dropout_rate: float = 0.2,
            mlp_dropout_rate: float = 0.2,
            post_norm: Optional[bool] = True
    ):
        '''
        ---------------
        Args:
            speech_hidden_size: hidden dimension of the output of speech encoder
            text_hidden_size: hidden dimension of LLM
            num_group_tokens: number of learnable grouping tokens
            attn_dropout_rate: dropout rate for nn.Dropout in the self attention layer
            mlp_dropout_rate: dropout rate for nn.Dropout in the MLP layer
            post_norm: whether to perform layer normalization at the end of the grouping layer
        '''
        super().__init__()
        self.num_group_tokens = num_group_tokens
        self.speech_hidden_size = speech_hidden_size
        self.text_hidden_size = text_hidden_size
        self.attn_dropout_rate = attn_dropout_rate
        self.mlp_dropout_rate = mlp_dropout_rate
        self.post_norm = post_norm

        self.selfattention = Attention(
            dim=self.speech_hidden_size,
            gumbel_softmax=False,
            attn_dropout_rate=self.attn_dropout_rate,
            mlp_dropout_rate=self.mlp_dropout_rate
        )

        self.crossattentionwithgumbel_softmax = Attention(
            dim=self.speech_hidden_size,
            gumbel_softmax=True,
            attn_dropout_rate=self.attn_dropout_rate,
            mlp_dropout_rate=self.mlp_dropout_rate            
        )
        
        self.projection = nn.Linear(self.speech_hidden_size, self.text_hidden_size)
        if self.post_norm:
            self.norm = nn.LayerNorm(normalized_shape=self.text_hidden_size)
        else:
            self.norm = nn.Identity()

    def _concat(self, group_tokens, x):
        '''
        ---------------
        Inputs:
            group_tokens: Trainable grouping tokens shape of [batch_size, num_group_tokens, hidden_size]
            x: [batch_size, seq_length, hidden_size]
        Outpus:
            return concatenation of [group_token, x] -> [batch_size, num_group_tokens + seq_length, hidden_size]
        '''
        return torch.concat([group_tokens, x], dim=1)
    
    def _split(self, x):
        return x[:, :self.num_group_tokens], x[:, self.num_group_tokens:]
        
    def forward(
            self,
            speech_features: torch.Tensor,
            gumbel_tau: float = 1.0,
            return_attn=False,
    ):
        '''
        -----------------------
        Inputs:
            speech_features: speech/dialect embedding shape of [batch_size, num_frames, speech_hidden_size]
            return_attn: bool value decide whether return attention matrix or not
        '''
        _, seq_length, _ = speech_features.shape

        kernel_size = seq_length // self.num_group_tokens
        pooling_layer = nn.AvgPool1d(kernel_size=kernel_size)
        self.group_tokens = torch.swapaxes(pooling_layer(speech_features.swapaxes(-1, -2)), -1, -2)
        
        input_tokens = self._concat(self.group_tokens, speech_features)
        update_input_tokens, _ = self.selfattention(input_tokens, input_tokens, input_tokens)
        group_tokens, _ = self._split(update_input_tokens)
        new_group_tokens, output_attn = self.crossattentionwithgumbel_softmax(
            group_tokens, 
            speech_features, 
            speech_features,
            gumbel_tau=gumbel_tau,
            return_attn=return_attn)
        new_group_tokens += group_tokens # [batch_size, num_group_tokens, speech_hidden_size]
        output = self.projection(new_group_tokens)
        output = self.norm(output)
        return output, output_attn


class globalAvgPool(nn.Module):
    def __init__(self, kernel_size: Optional[int] = None):
        '''
        Args:
            kernel_size: The size of kernel window used to take the average value
        '''
        super().__init__()
        self.kernel_size = kernel_size
    
    def forward(self, inputs: torch.Tensor):
        '''
        Inputs:
            inputs: [batch_size, seq_length, d_model]
        Outputs:
            avg_pool_output: [batch_size, d_model]
        '''
        _, seq_length, _ = inputs.shape
        
        if self.kernel_size is None:
            self.kernel_size = seq_length

        avg_inputs = torch.swapaxes(inputs, -1, -2)
        pool = nn.AvgPool1d(kernel_size=self.kernel_size)
        output = pool(avg_inputs)
        output = output.squeeze(dim=-1)
        return output


class DialectCLIPCausalLMOutput(CausalLMOutput):
    '''
    Base class for DialectCLIP casual language model(for conditional generation) outputs
    --------------------
    Args:
        loss (`torch.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next token prediction)
        logits (`torch.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    '''
    loss: Optional[torch.Tensor] = None
    logits: Optional[Union[tuple, torch.Tensor]] = None


class DialectCLIPPretrainedModel(PreTrainedModel):
    config_class = DialectCLIPConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, self.config.initializer_range)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, self.config.initializer_range)


class DialectCLIP(DialectCLIPPretrainedModel):
    def __init__(self, config: Optional[DialectCLIPConfig] = None):
        if config is None:
            config = DialectCLIPConfig()
        super().__init__(config=config)
        self.config = config

        # LLM initialize
        self.language_model_config = AutoConfig.from_pretrained(config.TEXT_MODEL_NAME)
        self.language_model = AutoModelForCausalLM.from_pretrained(config.TEXT_MODEL_NAME, config=self.language_model_config)
        self.lm_head = self.language_model.get_output_embeddings()
        for param in self.language_model.parameters():
            param.requires_grad = False
        
        # Speech Encoder initialize
        self.speech_config = AutoConfig.from_pretrained(config.SPEECH_MODEL_NAME)
        self.speech_model = AutoModel.from_pretrained(config.SPEECH_MODEL_NAME, config=self.speech_config)
        for param in self.speech_model.parameters():
            param.requires_grad = False

        # Grouping Layer initialize
        self.grouping_layer = GroupingLayer(
            speech_hidden_size=self.config.speech_dim,
            text_hidden_size=self.config.text_dim,
            num_group_tokens=self.config.num_group_tokens,
            attn_dropout_rate=self.config.attn_dropout_rate,
            mlp_dropout_rate=self.config.mlp_dropout_rate
        )

        # pooling layer for average pooling feature
        self.pool_speech = globalAvgPool()
        self.pool_language = globalAvgPool()

        # logit_scale
        self.logit_scale = nn.Parameter(torch.tensor(config.logit_scale_init_value, dtype=torch.float32))

    @torch.no_grad()
    def encode_prompt(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.Tensor]] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = True,
            return_dict: Optional[bool] = None,
            pool: Optional[bool] = None
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        outputs = self.language_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds = inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_states = outputs[0]
        logits = self.lm_head(last_hidden_states)
        if pool:
            pool_embedding = self.pool_language(last_hidden_states)
            pool_embedding = F.normalize(pool_embedding, p=2, dim=1)
        return (logits, pool_embedding) if pool is not None else (logits,)
    
    def encode_speech(
            self,
            input_features: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            pool: Optional[bool] = None         
    ):
        '''
        Inputs:
            input_features: torch.Tensor, shape of [batch_size, num_mels, sequence_length]: \
                Float values of mel features extracted from the raw speech waveform.
            head_mask: torch.Tensor, shape of [encoder_layers, encoder_attention_heads]: \
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:
                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            pool: whether perform average pooling or not
        '''
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if input_features is None:
            raise RuntimeError(f"input_features should be given, but got None")
        
        outputs = self.speech_model.encoder(
            input_features=input_features,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict                
        )
        last_hidden_states = outputs[0]
        last_hidden_states, _ = self.grouping_layer(last_hidden_states, self.config.tau, self.config.output_attentions)
        if pool:
            pool_embedding = self.pool_speech(last_hidden_states)
            pool_embedding = F.normalize(pool_embedding, p=2, dim=1)
            return last_hidden_states, pool_embedding
        else:
            return last_hidden_states
            

class DialectCLIPForConditionalGeneration(DialectCLIP):
    def __init__(self, config: Optional[DialectCLIPConfig] = None):
        super().__init__(config=config)
        if config is None:
            config = DialectCLIPConfig()
        self.config = config
        self.embedding_layer = self.language_model.get_input_embeddings()
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def get_input_embeddings(self):
        return self.embedding_layer
    
    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)
    
    def _merge_input_ids_with_speech_features(self, speech_features, inputs_embeds, input_ids, attention_mask, labels):
        '''
        Merge speech embedding with text inputs as LLM inputs
        ---------------------------------
        Inputs:
            speech_features: speech/dialect embedding generated from speech encoder shape of [batch_size, speech_seq_length, d_dimension]
            inputs_embeds: word embedding generated by nn.Embedding Layer shape of [batch_size, text_seq_length, d_dimension]
            input_ids: [batch_size, text_seq_length] tokens id tokenized by pretrained tokenizer
            attention_mask: [batch_size, text_seq_length] attention mask
            labels: [batch_size, text_seq_length] compute loss only on the position of text, default ignore_ids = -100
        '''
        batch_size, speech_seq_length, d_model = speech_features.shape
        _, text_seq_length = input_ids.shape
        left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(self.pad_token_id))
        # create a mask to know where special speech tokens <speech> are
        special_speech_token_mask = input_ids == self.config.speech_token_index
        num_special_speech_tokens_per_seq = torch.sum(special_speech_token_mask, dim=-1)
        max_seq_length = num_special_speech_tokens_per_seq.max().item() * (speech_seq_length - 1) + text_seq_length
        indices = torch.where(input_ids != self.config.speech_token_index)
        batch_indices = indices[0]
        non_speech_indices = indices[1]

        # calculate new text token position in merged speech-text embedding sequence
        new_token_positions = torch.cumsum((special_speech_token_mask * (speech_seq_length - 1) + 1), -1) - 1
        nb_speech_token = max_seq_length - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_speech_token[:, None]
        text_to_overwrite = new_token_positions[batch_indices, non_speech_indices]
        text_embedding_position = (batch_indices, text_to_overwrite)

        # create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_seq_length, d_model, dtype=inputs_embeds.dtype,
            device=self.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_seq_length, dtype=attention_mask.dtype,
            device=self.device
        )

        if labels is not None:
            final_labels = torch.full_like(final_attention_mask, self.config.ignore_index, device=self.device)

        # fill the final embedding corresponding to the original text embedding
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_speech_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_speech_indices]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_speech_indices]
        
        # fill the final embedding corresponding to the speech features
        speech_to_overwrite = torch.full(
            (batch_size, max_seq_length), True, dtype=torch.bool, device=self.device
        )
        speech_to_overwrite[batch_indices, text_to_overwrite] = False
        speech_to_overwrite &= speech_to_overwrite.cumsum(-1) - 1 >= nb_speech_token[:, None].to(device=self.device)

        if speech_to_overwrite.sum() != speech_features.shape[:-1].numel():
            raise ValueError(f"{speech_to_overwrite.sum()} not equal to {speech_features.shape[:-1].numel()}, Probably you should add an <AUDIO> token into the prompt to indicate the position of audio signal")
        
        final_embedding[speech_to_overwrite] = speech_features.view(-1, d_model)
        final_attention_mask |= speech_to_overwrite
        position_ids = torch.masked_fill(final_attention_mask.cumsum(-1) - 1, final_attention_mask == 0, 1)

        if self.pad_token_id is not None:
            indices = torch.where(input_ids == self.pad_token_id)
            batch_indices = indices[0]
            pad_indices = indices[1]
            indices_to_mask = new_token_positions[batch_indices, pad_indices]

            final_embedding[batch_indices, indices_to_mask] = 0

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels, position_ids, text_embedding_position
    

    def forward_fn(
            self,
            input_ids: Optional[torch.Tensor] = None,
            input_features: Optional[torch.Tensor] = None, 
            attention_mask: Optional[torch.Tensor] = None, 
            position_ids: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None, 
    ):
        '''
        ----------------------
        Inputs:
            input_ids: token id shape of [batch_size, text_seq_length]
            input_features: Inputs of speech encoder shape of [batch_size, speech_seq_length, num_mels]        
            attention_mask: attention mask for batch sequence
            inputs_embeds: output of nn.Embedding(config.vocab_size, hidden_size), shape of [batch_size, seq_length, hidden_size]
            labels: for next token prediction task which is actually the same as input_ids            
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        '''
        encode_features, pool_encode_features = self.encode_speech(
            input_features=input_features,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pool=True
        )

        batch_size, _, hidden_size = encode_features.shape
        inputs_embeds = self.embedding_layer(input_ids)

        inputs_embeds, attention_mask, labels, position_ids, text_embedding_position \
            = self._merge_input_ids_with_speech_features(encode_features, inputs_embeds=inputs_embeds, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        outputs = self.language_model.model(
            input_ids=None,
            attention_mask = attention_mask,
            position_ids = position_ids,
            inputs_embeds = inputs_embeds,
            use_cache = use_cache,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict
        )

        last_hidden_states = outputs[0]
        logits = self.lm_head(last_hidden_states)
        logits = logits.float()
        
        loss_casuallm = casuallm_loss(
            logits=logits,
            labels=labels,
            attention_mask=attention_mask,
            ignore_index=self.config.ignore_index
        )

        batch_indices, text_to_overwrite = text_embedding_position
        text_embedding = last_hidden_states[batch_indices, text_to_overwrite].view(batch_size, -1, hidden_size)

        text_pool_features = self.pool_language(text_embedding)
        text_pool_features = F.normalize(text_pool_features, p=2, dim=1)

        logits_speech_text = self.logit_scale *  pool_encode_features @ text_pool_features.t()
        # shape = [batch_size, batch_size]

        loss_clip = dialect_clip_loss(similarity=logits_speech_text, device=self.device)

        return loss_clip, loss_casuallm, pool_encode_features
    

    def train_forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            input_speech_features: Optional[torch.Tensor] = None,
            input_dialect_features: Optional[torch.Tensor] = None,   
            attention_mask: Optional[torch.Tensor] = None, 
            position_ids: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,             
    ):
        '''
        ----------------------
        Inputs:
            input_ids: token id shape of [batch_size, text_seq_length]
            input_speech_features: Speech inputs of speech encoder shape of [batch_size, speech_seq_length, num_mels]
            input_dialect_features: Dialect inputs of speech encoder shape of [batch_size, speech_seq_length, num_mels]            
            attention_mask: attention mask for batch sequence
            inputs_embeds: output of nn.Embedding(config.vocab_size, hidden_size), shape of [batch_size, seq_length, hidden_size]
            labels: for next token prediction task which is actually the same as input_ids            
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        '''
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # speech forward
        loss_clip_speech, loss_casuallm_speech, speech_pool_features = self.forward_fn(
            input_ids=input_ids,
            input_features=input_speech_features, 
            attention_mask=attention_mask, 
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,              
        )

        # dialect_forward
        loss_clip_dialect, loss_casuallm_dialect, dialect_pool_features = self.forward_fn(
            input_ids=input_ids,
            input_features=input_dialect_features, 
            attention_mask=attention_mask, 
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,              
        )

        logits_speech_text = self.logit_scale * speech_pool_features @ dialect_pool_features.t() # shape = [batch_size, batch_size]
        loss_clip_speech_dialect = dialect_clip_loss(similarity=logits_speech_text, device=self.device)
        loss_clip_dialect += loss_clip_speech_dialect

        loss_clip = loss_clip_speech + loss_clip_dialect
        loss_casuallm = loss_casuallm_speech + loss_casuallm_dialect

        loss = self.config.alpha * loss_clip + self.config.beta * loss_casuallm

        return loss

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            input_features: Optional[torch.Tensor] = None,   
            attention_mask: Optional[torch.Tensor] = None, 
            position_ids: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,            
    ):
        '''
        ----------------------
        Inputs:
            input_ids: token id shape of [batch_size, text_seq_length]
            input_features: Inputs of speech encoder shape of [batch_size, speech_seq_length, num_mels] 
            attention_mask: attention mask for batch sequence
            inputs_embeds: output of nn.Embedding(config.vocab_size, hidden_size), shape of [batch_size, seq_length, hidden_size]
            labels: for next token prediction task which is actually the same as input_ids            
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        '''
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if input_features is None:
            raise ValueError("speech signal not provided")

        speech_features = self.encode_speech(
            input_features=input_features,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if inputs_embeds is None:
            inputs_embeds = self.embedding_layer(input_ids)

            inputs_embeds, attention_mask, labels, position_ids, _\
                = self._merge_input_ids_with_speech_features(speech_features, inputs_embeds=inputs_embeds, input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        outputs = self.language_model.model(
            input_ids=None,
            attention_mask = attention_mask,
            position_ids = position_ids,
            inputs_embeds = inputs_embeds,
            use_cache = use_cache,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict
        )

        last_hidden_states = outputs[0]
        logits = self.lm_head(last_hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1), ignore_index=self.config.ignore_index
            )

        if not return_dict:
            return (loss,) + logits if loss is not None else logits
        
        return DialectCLIPCausalLMOutput(
            loss=loss,
            logits=logits
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs
    ):
        if inputs_embeds is not None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        model_inputs.update(
            {
                "input_features": input_features,
                "attention_mask": attention_mask,
                "use_cache": kwargs.get("use_cache")
            }
        )
        return model_inputs
