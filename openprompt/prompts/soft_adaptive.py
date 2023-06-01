
import os

from torch.nn.parameter import Parameter
from openprompt.utils.logging import logger



from openprompt.data_utils import InputExample, InputFeatures
from typing import *

from transformers import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from openprompt import Template
from openprompt.prompts import ManualTemplate, ManualVerbalizer
#from gumbel import gumbel_softmax

import torch
from torch import nn
import os, sys
sys.path.append(os.path.abspath('/nfs/users/weicheng/textCL/OpenPrompt/tutorial'))
import global_var
class SoftTemplate2(Template):
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
                ):
        super().__init__(tokenizer=tokenizer,
                         placeholder_mapping=placeholder_mapping)
        self.raw_embedding = model.get_input_embeddings()
        self.raw_embedding.requires_grad_(False)
        self.model_is_encoder_decoder = model.config.is_encoder_decoder
        self.random_range = random_range
        self.num_tokens = num_tokens
        self.initialize_from_vocab = initialize_from_vocab

        self.text = text
        # self.default_text1 = {"placeholder<text_a> <mask>"
        # self.default_text2 = "<text_a> <text_b> <mask>".split()

        if soft_embeds is not None:
            self.soft_embeds = soft_embeds
            self.num_tokens = len(soft_embeds)
        else:
            if self.num_tokens>0:
                self.generate_parameters()


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
        position_logit_ini = torch.FloatTensor(self.num_tokens+1, 1).uniform_(-self.random_range, self.random_range)#, 0)
        self.soft_embeds = nn.Parameter(soft_embeds.cuda(), requires_grad=True)
        self.position_emb = nn.Parameter(position_logit_ini.cuda(), requires_grad=True)



    # def assemble_soft_template(self, soft_embeds, inputs_embeds, dim, logit):
    #     # res = []
    #     # if self.idx == 0:
    #     #     res = torch.cat([inputs_embeds, soft_embeds], dim)
    #     # elif self.idx == self.num_tokens:
    #     #     res = torch.cat([soft_embeds, inputs_embeds], dim)
    #     # else:
    #     #     res = torch.cat([soft_embeds[:, 0:self.idx], inputs_embeds], dim)
    #     #     res = torch.cat([res, soft_embeds[:, self.idx:]], dim)
    #
    #     res = torch.cat([inputs_embeds, soft_embeds], dim)
    #     output = logit[0] * torch.roll(res, 0, 1)
    #     for i in range(1, self.num_tokens+1):
    #         output = output + logit[i] * torch.roll(res, i, 1)
    #
    #     return output

    def assemble_soft_template(self, soft_embeds, inputs_embeds, dim, logit, eos_position):

        res = torch.cat([soft_embeds, inputs_embeds], dim)
        for i in range(res.size(dim=0)):
            if res.dim() == 3:
                #print(logit.dim(),'\n')
                output = logit[i, 0] * torch.roll(res[i, :, :], 0, 0)
                for j in range(1, self.num_tokens + 1):
                    output = output + logit[i, j] * torch.cat(
                        [torch.cat([torch.cat([soft_embeds[i, j:, :], inputs_embeds[i, :eos_position[i], :]], dim=0) \
                                       , soft_embeds[i, :j, :]], dim=0), \
                         inputs_embeds[i, eos_position[i]:, :]], dim=0)
                res[i, :, :] = output  #
            else:
                output = logit[i, 0] * torch.roll(res[i, :], 0, 0)
                for j in range(1, self.num_tokens + 1):  # torch.roll(res[i, :], j, 0)
                    output = output + logit[i, j] * torch.cat(
                        [torch.cat([torch.cat([soft_embeds[i, j:], inputs_embeds[i, :eos_position[i]]], dim=0) \
                                       , soft_embeds[i, :j]], dim=0), \
                         inputs_embeds[i, eos_position[i]:]], dim=0)
                res[i, :] = output
        return res

    def process_batch(self, batch: Union[Dict, InputFeatures]) -> Union[Dict, InputFeatures]:
        """
        Convert input_ids to inputs_embeds
        for normal tokens, use the embedding layer of PLM
        for soft tokens, use a new embedding layer which is initialized with their corresponding embedding of hard tokens
        """
        temperature = global_var.get_value('temperature')
        # print("temperature: \n", temperature)
        logit = torch.nn.functional.gumbel_softmax(self.position_emb, temperature, True, dim=0) #torch.softmax(self.position_emb,0)#


        #self.idx = torch.argmax(logit)

        inputs_embeds = self.raw_embedding(batch['input_ids'])
        batch_size = inputs_embeds.size(0)
        logit_bt = logit.repeat(batch_size, 1, 1)#.squeeze()
        eos_position = [torch.nonzero(batch['attention_mask'][i])[-1][0] for i in
                        range(batch['attention_mask'].size(0))]

        if self.num_tokens>0:
            soft_embeds = self.soft_embeds.repeat(batch_size, 1, 1)
            inputs_embeds = self.assemble_soft_template(soft_embeds, inputs_embeds, 1, logit_bt, eos_position)

        batch['input_ids'] = None
        batch['inputs_embeds'] = inputs_embeds
        if 'attention_mask' in batch and self.num_tokens>0:
            am = batch['attention_mask']
            logit_detach = logit_bt.detach()
            batch['attention_mask'] = self.assemble_soft_template(torch.ones((batch_size,self.num_tokens), dtype = am.dtype,device=am.device), am, -1, logit_detach, eos_position)
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
