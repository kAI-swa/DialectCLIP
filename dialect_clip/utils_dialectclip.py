import jiwer
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Optional, Union

class WER:
    def __init__(self, transforms: Optional[jiwer.Compose] = None):
        self.transforms = transforms
        if transforms is None:
            self.transforms = jiwer.Compose([
                jiwer.RemoveEmptyStrings(),
                jiwer.ToLowerCase(),
                jiwer.RemoveMultipleSpaces(),
                jiwer.Strip(),
                jiwer.RemovePunctuation(),
                jiwer.ReduceToListOfListOfWords()
                ]
            )

    def compute(self, reference: Union[str, list], hypothesis: Union[str, list]):
        wer = jiwer.wer(
            reference=reference,
            hypothesis=hypothesis,
            truth_transform=self.transforms,
            hypothesis_transform=self.transforms
        )
        return wer


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    '''
    Inputs: 
        logits: [batch_size, batch_size], similarity among pairs
    Outputs:
        Cross Entropy loss within a batch for one modality
    '''
    target=torch.arange(len(logits))
    return F.cross_entropy(input=logits, target=target)


def dialect_clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    '''
    Inputs: 
        similarity: [batch_size, batch_size], similarity score among pairs
    Outputs:
        dialect_CLIP loss
    '''
    loss_1 = contrastive_loss(similarity)
    loss_2 = contrastive_loss(similarity.t())
    return (loss_1 + loss_2) / 2


def casuallm_loss(
        self,
        logits: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
):
    loss_casuallm = None
    if labels is not None:
        if attention_mask is not None:
            shift_attention_mask = attention_mask[..., 1:]
            shift_logits = logits[..., :-1, :][shift_attention_mask != 0].contiguous()
            shift_labels = labels[..., 1:][shift_attention_mask != 0].contiguous()
        else:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_casuallm = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1), ignore_index=self.config.ignore_idx
        )
    else:
        raise ValueError("Labels should be given to compute language modeling loss")
    return loss_casuallm


def line_plot(y_list, title, *args, **kwargs):
    plt.figure(dpi=600)
    plt.title(f"{title} Curve")
    x = range(len(y_list))
    plt.plot(x, y_list, 'r-')
    plt.ylabel(f'{title}')
    plt.xlabel('Steps')
    plt.savefig(f"./Figure/{title}.png")
