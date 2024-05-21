import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig, WhisperForConditionalGeneration, AutoProcessor
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput, Seq2SeqLMOutput
from transformers.generation import GenerationConfig
from peft import LoraConfig, get_peft_model
from typing import Optional, Tuple, Union
from .configuration_dialectclip import DialectCLIPConfig
from .utils_dialectclip import dialect_clip_loss


# Copied from transformers.models.bart.modeling_bart.shift_tokens_right
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


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


class DialectCLIPSeq2SeqLMOutput(Seq2SeqLMOutput):
    """
    Base class for DialectCLIP sequence-to-sequence language models outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_pool_last_hidden_state: Optional[torch.FloatTensor] = None    


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
        
        # Speech Model initialize
        self.speech_model = WhisperForConditionalGeneration.from_pretrained(config.SPEECH_MODEL_NAME)

        # pooling layer for average pooling feature
        self.pool_speech = globalAvgPool()

        # logit_scale
        self.logit_scale = nn.Parameter(torch.tensor(config.logit_scale_init_value, dtype=torch.float32))

    def get_speech_encoder(self):
        return self.speech_model.get_encoder()

    def get_speech_decoder(self):
        return self.speech_model.get_decoder()

    def get_speech_output_embeddings(self):
        return self.speech_model.get_output_embeddings()

    def set_speech_output_embeddings(self, new_embeddings):
        self.speech_model.proj_out = new_embeddings

    def get_speech_input_embeddings(self) -> nn.Module:
        return self.speech_model.get_input_embeddings()

    def freeze_speech_encoder(self):
        """
        Calling this function will disable the gradient computation for the Whisper encoder so that its parameters will
        not be updated during training.
        """
        self.speech_model.encoder._freeze_parameters()

    def train_forward(
            self,
            input_speech_features: Optional[torch.FloatTensor] = None,
            input_dialect_features: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
            decoder_position_ids: Optional[Tuple[torch.LongTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,            
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            pool: Optional[bool] = None             
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

        if labels is None:
            raise ValueError("Please provide labels to train")

        # speech forward
        speech_outputs = self.forward(
            input_speech_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_position_ids=decoder_position_ids,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pool=pool
        )
        loss_casuallm_speech = speech_outputs[0]
        speech_pool_features = speech_outputs[-1]

        # dialect_forward
        dialect_outputs = self.forward(
            input_dialect_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_position_ids=decoder_position_ids,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,     
            pool=pool         
        )
        loss_casuallm_dialect = dialect_outputs[0]
        dialect_pool_features = dialect_outputs[-1]

        logits_speech_text = self.logit_scale * speech_pool_features @ dialect_pool_features.t() # shape = [batch_size, batch_size]
        loss_clip = dialect_clip_loss(similarity=logits_speech_text, device=self.device)
        loss_casuallm = loss_casuallm_speech + loss_casuallm_dialect

        loss = self.config.alpha * loss_clip + self.config.beta * loss_casuallm
        return loss

    def forward(
            self,
            input_features: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
            decoder_position_ids: Optional[Tuple[torch.LongTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,            
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
        
        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.speech_model.config.pad_token_id, self.speech_model.config.decoder_start_token_id
                )
        
        outputs = self.speech_model(
            input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_position_ids=decoder_position_ids,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_last_hidden_states = outputs[-3]
        if pool:
            pool_features = self.pool_speech(encoder_last_hidden_states)
            pool_features = F.normalize(pool_features, p=2, dim=1)

        if not return_dict:
            return (outputs + (pool_features,)) if pool_features is not None else outputs
    
        return DialectCLIPSeq2SeqLMOutput(
            loss=outputs.loss,
            logits=outputs.lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            encoder_last_hidden_states=pool_features     
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        use_cache=None,
        encoder_outputs=None,
        decoder_attention_mask=None,
        **kwargs,
    ):
        decoder_position_ids = None
        if decoder_attention_mask is not None:
            decoder_position_ids = (decoder_attention_mask.cumsum(-1) - 1).clamp(min=0)

        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

            if decoder_position_ids is not None and decoder_position_ids.shape[1] > decoder_input_ids.shape[1]:
                decoder_position_ids = decoder_position_ids[:, remove_prefix_length:]

        return {
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "use_cache": use_cache,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_position_ids": decoder_position_ids,
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
            

class DialectCLIPWithGER(DialectCLIP):
    def __init__(self, config: Optional[DialectCLIPConfig] = None):
        super().__init__(config=config)
        if config is None:
            config = DialectCLIPConfig()
        self.config = config

        # LLM initialize
        self.language_model_config = AutoConfig.from_pretrained(config.TEXT_MODEL_NAME)
        self.language_model = AutoModelForCausalLM.from_pretrained(config.TEXT_MODEL_NAME, config=self.language_model_config)
        for param in self.language_model.parameters():
            param.requires_grad = False

        # processor
        self.processor = AutoProcessor.from_pretrained(self.config.SPEECH_MODEL_NAME)

        # LoRA configuration
        self.peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            target_modules=self.config.target_modules
        )
        self.language_model = get_peft_model(model=self.language_model, peft_config=self.peft_config)

    def forward(
            self,
            input_features: Optional[torch.FloatTensor] = None,           
    ):
        generation_config = GenerationConfig(
            do_sample=self.config.do_sample,
            num_beams=self.config.num_beams,
            max_length=self.config.max_length
        )    
        generate_ids = super().generate(inputs=input_features, generation_config=generation_config)
        hypothesis_list = self.processor.batch_decode(generate_ids, skip_special_tokens=True)[0]

