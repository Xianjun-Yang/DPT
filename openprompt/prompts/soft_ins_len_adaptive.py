
import os
from torch.nn.parameter import Parameter
from openprompt.utils.logging import logger
from openprompt.data_utils import InputExample, InputFeatures
from openprompt.utils import round_list, signature
from typing import *

from transformers import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from openprompt import Template
from openprompt.prompts import ManualTemplate, ManualVerbalizer

import torch
from torch import nn
import os, sys
#fine_path = '/nfs/users/xianjun/'
file_path = '../../../../../Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/5831d4c18cacd141373ae04b4e1a42a6/Message/MessageTemp/'  #'/home/dsi/xyang'
sys.path.append(os.path.abspath( file_path + '/tutorial')) # /project/OpenPrompt
import global_var

class MLP_pos(nn.Module):
    '''
      Multilayer Perceptron.
    '''

    def __init__(self, dim_size, num_soft_token):
        super().__init__()
        self.layers = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(dim_size, num_soft_token + 1),
            nn.ReLU(),
        )

    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)

class MLP_len(nn.Module):
    '''
      Multilayer Perceptron.
    '''

    def __init__(self, dim_size, num_soft_token):
        super().__init__()
        self.layers = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(dim_size, num_soft_token + 1),
            nn.ReLU(),
        )

    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)

class SoftTemplate_ins_len(Template):
    r"""This is the implementation of `The Power of Scale for Parameter-Efficient
    Prompt Tuning <https://arxiv.org/pdf/2104.08691v1.pdf>`_ . Similar to :obj:`PrefixTuningTemplate`,
    This template also does not need any textual template. Addition tokens are directly
    concatenated into the input ids. There are two initializations of the new tokens.
    (1). random initialization. (2) initialize with the tokens of the plm (We simply take
    the first n_tokens similar to their implementation).
    """
    registered_inputflag_names = ["loss_ids", "shortenable_ids"]

    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 text: Optional[str] = None,
                 soft_embeds: Optional[torch.FloatTensor] = None,
                 num_tokens: int=20,
                 initialize_from_vocab: Optional[bool] = True,
                 random_range: Optional[float] = 0.5,
                 placeholder_mapping: dict = {'<text_a>':'text_a','<text_b>':'text_b'},
                 pos_idx = False,
                 len_idx = False,
                ):
        super().__init__(tokenizer=tokenizer,
                         placeholder_mapping=placeholder_mapping)
        self.plm = model
        self.raw_embedding = model.get_input_embeddings()
        self.raw_embedding.requires_grad_(False)
        self.model_is_encoder_decoder = model.config.is_encoder_decoder
        self.random_range = random_range
        self.num_tokens = num_tokens
        self.initialize_from_vocab = initialize_from_vocab

        self.text = text
        self.pos_idx = pos_idx
        self.len_idx = len_idx
        # self.default_text1 = {"placeholder<text_a> <mask>"
        # self.default_text2 = "<text_a> <text_b> <mask>".split()

        if soft_embeds is not None:
            self.soft_embeds = soft_embeds
            self.num_tokens = len(soft_embeds)
        else:
            if self.num_tokens>0:
                self.generate_parameters() # generate soft_embeds
                self.pos_transform = MLP_pos(self.raw_embedding.weight.size(1), self.num_tokens) # share between all tasks
                self.len_transform = MLP_len(self.raw_embedding.weight.size(1), self.num_tokens) # share between all tasks

    def on_text_set(self):
        self.text = self.parse_text(self.text)

    def wrap_one_example(self, example) -> List[Dict]:  #TODO this automatic generated template may not be able to process diverse data format.
        if self.text is None:
            logger.warning("You didn't provide text template for softprompt. Using default template, is this intended?")
            if example.text_b is None:
                self.text = self.default_text1
            else:
                self.text = self.default_text2
        return super().wrap_one_example(example)

    def generate_parameters(self) -> None:
        """
        generate parameters needed for soft tokens embedding in soft-prompt
        for soft tokens, use a new embedding layer which is initialized with their corresponding embedding of hard tokens
        """
        if self.initialize_from_vocab:
            soft_embeds = self.raw_embedding.weight[:self.num_tokens].clone().detach()
        else:
            soft_embeds = torch.FloatTensor(self.num_tokens, self.raw_embedding.weight.size(1)).uniform_(-self.random_range, self.random_range)

        self.soft_embeds = nn.Parameter(soft_embeds, requires_grad=True) # fix the embedding of soft tokens for all instances

    def assemble_soft_template(self, soft_embeds, inputs_embeds, dim, logit, eos_position):
        if self.pos_idx: # fixed position
            res = []
            self.idx = torch.tensor([0]).to(soft_embeds)
            if self.idx == 0:
                res = torch.cat([soft_embeds, inputs_embeds], dim)
        else: # dynamic position
            batch_size = inputs_embeds.size(dim=0)
            res = torch.cat([soft_embeds[0:batch_size, :], inputs_embeds ], dim)
            for i in range(res.size(dim=0)):
                if res.dim() == 3:
                    output = logit[i, 0] * torch.cat([soft_embeds[0:batch_size, :, :], inputs_embeds], dim)[i]
                    for j in range(1, self.num_tokens + 1):
                        output = output + logit[i, j] * torch.cat([soft_embeds[j*batch_size: (j+1)*batch_size, :, :], inputs_embeds], dim)[i]
                    res[i, :, :] = output
                else:
                    output = logit[i, 0] * torch.cat([soft_embeds[0:batch_size, :], inputs_embeds], dim)[i]
                    for j in range(1, self.num_tokens + 1):
                        output = output + logit[i, j] * torch.cat([soft_embeds[j*batch_size: (j+1)*batch_size, :], inputs_embeds], dim)[i]
                    res[i, :] = output
        return res

    def process_batch_pre(self, batch: Union[Dict, InputFeatures]) -> Union[Dict, InputFeatures]:
        """
        Convert input_ids to inputs_embeds
        for normal tokens, use the embedding layer of PLM
        for soft tokens, use a new embedding layer which is initialized with their corresponding embedding of hard tokens
        """
        inputs_embeds = self.raw_embedding(batch['input_ids'])
        batch_size = inputs_embeds.size(0)
        batch['input_ids'] = None
        batch['inputs_embeds'] = inputs_embeds

        return batch

    def process_batch(self, batch: Union[Dict, InputFeatures]) -> Union[Dict, InputFeatures]:
        """
        Convert input_ids to inputs_embeds
        for normal tokens, use the embedding layer of PLM
        for soft tokens, use a new embedding layer which is initialized with their corresponding embedding of hard tokens
        """
        input_batch = self.process_batch_pre(batch)
        forward_keys = signature(self.plm.forward).args
        input_batch = {key: input_batch[key] for key in batch if key in forward_keys}
        outputs = self.plm(**input_batch, output_hidden_states=True)
        outputs = self.post_processing_outputs(outputs)
        last_state = outputs['encoder_last_hidden_state']
        pooled_sentence = torch.mean(last_state, dim=1)

        temperature = global_var.get_value('temperature')

        position_emb = self.pos_transform(pooled_sentence)
        position_emb_len = self.len_transform(pooled_sentence)
        logit = torch.nn.functional.gumbel_softmax(position_emb, temperature, True, dim=1)
        logit_len = torch.nn.functional.gumbel_softmax(position_emb_len, temperature, True, dim=1)

        inputs_embeds = batch['inputs_embeds']
        batch_size = inputs_embeds.size(0)
        eos_position = [torch.nonzero(batch['attention_mask'][i])[-1][0] for i in range(batch['attention_mask'].size(0))]
        if self.num_tokens>0:
            soft_embeds = self.soft_embeds.repeat(batch_size, 1, 1)
            soft_embeds_len = soft_embeds.repeat(self.num_tokens + 1, 1, 1)
            for j in range(0, self.num_tokens + 1):
                soft_embeds_len[j*batch_size: (j+1)*batch_size, :, :] = torch.cat( [soft_embeds[:, :j, :], -1e20*torch.ones_like(soft_embeds[:, j:, :]).to(soft_embeds)], dim=1)

            inputs_embeds = self.assemble_soft_template(soft_embeds_len, inputs_embeds, 1, logit_len, eos_position)

        batch['input_ids'] = None
        batch['inputs_embeds'] = inputs_embeds
        if 'attention_mask' in batch and self.num_tokens>0:
            am = batch['attention_mask']
            logit_detach = logit_len.detach()
            batch['attention_mask'] = self.assemble_soft_template(torch.ones((batch_size*(self.num_tokens+1), self.num_tokens),\
                                                                     dtype = am.dtype,device=am.device), am, -1, logit_detach, eos_position)
        return batch


    def post_processing_outputs(self, outputs: torch.Tensor):
        r"""Post processing the outputs of language models according
        to the need of template. Most templates don't need post processing,
        The template like SoftTemplate, which appends soft template as a module
        (rather than a sequence of input tokens) to the input,
        should remove the outputs on these positions to keep the seq_len the same
        """
        if not self.model_is_encoder_decoder:
            outputs.logits = outputs.logits[:, self.num_tokens:,: ]
        return outputs

