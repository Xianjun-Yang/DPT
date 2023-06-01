from tqdm import tqdm
#from openprompt.data_utils import PROCESSORS
import os, sys
sys.path.append(os.path.abspath('/nfs/users/weicheng/textCL/OpenPrompt/openprompt'))
from huggingface_dataset2 import PROCESSORS
from openprompt.data_utils.huggingface_dataset import SuperglueMultiRCProcessor, SuperglueBoolQProcessor, SuperglueCBProcessor, SuperglueCOPAProcessor, SuperglueRTEProcessor, SuperglueWiCProcessor, SuperglueWSCProcessor, SuperglueRecordProcessor, YahooAnswersTopicsProcessor
import argparse

parser = argparse.ArgumentParser("")

parser.add_argument("--gpu", type=str, default="3")
parser.add_argument("--aneal", type=float, default=10.0)
parser.add_argument("--shot", type=int, default=32)
parser.add_argument("--seed", type=int, default=144)
parser.add_argument("--multigpu", type=bool, default=False)
parser.add_argument("--plm_eval_mode", action="store_true", help="whether to turn off the dropout in the freezed model. Set to true to turn off.")
parser.add_argument("--tune_plm", action="store_true")
parser.add_argument("--model", type=str, default='t5-lm', help="We test both t5 and t5-lm in this scripts, the corresponding tokenizerwrapper will be automatically loaded.")
parser.add_argument("--model_name_or_path", default='/nfs/users/weicheng/textCL/OpenPrompt/plm/t5-large-lm-adapt/')
parser.add_argument("--project_root", default="/nfs/users/weicheng/textCL/OpenPrompt/", help="The project root in the file system, i.e. the absolute path of OpenPrompt")
parser.add_argument("--template_id", type=int, default=0)
parser.add_argument("--verbalizer_id", type=int, default=0)
parser.add_argument("--data_dir", type=str, default="/nfs/users/weicheng/textCL/OpenPrompt/data_dir") # sometimes, huggingface datasets can not be automatically downloaded due to network issue, please refer to 0_basic.py line 15 for solutions.
parser.add_argument("--dataset",type=str, default="boolq")
parser.add_argument("--result_file", type=str, default="../results.txt")
parser.add_argument("--max_steps", default=20000, type=int)
parser.add_argument("--prompt_lr", type=float, default=0.3)
parser.add_argument("--warmup_step_prompt", type=int, default=500)
parser.add_argument("--init_from_vocab", type=bool, default=True)
parser.add_argument("--eval_every_steps", type=int, default=500)
parser.add_argument("--soft_token_num", type=int, default=20)
parser.add_argument("--optimizer", type=str, default="Adafactor")
args = parser.parse_args()

gpu_list = list(args.gpu)
gpu_list_str = ','.join(map(str, gpu_list))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_list_str)



import torch

#device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


from openprompt.data_utils.utils import InputExample

import numpy as np

# os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3'





# from accelerate import infer_auto_device_map

from openprompt import PromptDataLoader
from openprompt.prompts import ManualVerbalizer
import sys
from torch import nn
sys.path.append('/nfs/users/weicheng/textCL/OpenPrompt/openprompt/prompts')
sys.path.append(os.getcwd())
print(sys.path)
import soft_ins_pos_adaptive
import global_var
from soft_ins_pos_adaptive import SoftTemplate_ins_pos
from openprompt import PromptForClassification
import time
import os

from torch.utils.tensorboard import SummaryWriter
#
# import os
# os.environ['LD_LIBRARY_PATH'] = "/usr/local/cuda/lib64/"




args.result_file = os.path.join(args.project_root, args.result_file)

content_write = "="*20+"\n"
content_write += f"dataset {args.dataset}\t"
content_write += f"temp {args.template_id}\t"
content_write += f"verb {args.verbalizer_id}\t"
content_write += f"model {args.model}\t"
content_write += f"seed {args.seed}\t"
content_write += f"shot {args.shot}\t"
content_write += f"plm_eval_mode {args.plm_eval_mode}\t"
content_write += f"init_from_vocab {args.init_from_vocab}\t"
content_write += f"eval_every_steps {args.eval_every_steps}\t"
content_write += f"prompt_lr {args.prompt_lr}\t"
content_write += f"optimizer {args.optimizer}\t"
content_write += f"warmup_step_prompt {args.warmup_step_prompt}\t"
content_write += f"soft_token_num {args.soft_token_num}\t"
content_write += "\n"

print(content_write)

import random
this_run_unicode = str(random.randint(0, 1e10))

from openprompt.utils.reproduciblity import set_seed
set_seed(args.seed)

# use lm-adapted version or t5-v1.1 checkpoint. Note that the original t5 checkpoint has been pretrained
# on part of GLUE dataset, thus should not be used.
from openprompt.plms.seq2seq import T5TokenizerWrapper, T5LMTokenizerWrapper
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration
from openprompt.data_utils.data_sampler import FewShotSampler
from openprompt.plms import load_plm

plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model_name_or_path)
dataset = {}
temperature = 1.0
global_var._init()
global_var.set_value('temperature', 1.0)

# Below are multiple dataset examples, including few-shot ones.
if args.dataset == "boolq":
    Processor = PROCESSORS["super_glue.boolq"]
    proce = Processor()
    dataset['train'] = proce.get_examples(args.data_dir, "train")#get_train_examples(args.data_dir)
    dataset['validation'] = proce.get_examples(args.data_dir, "dev")#get_dev_examples(args.data_dir)
    dataset['test'] = proce.get_examples(args.data_dir, "test")#get_test_examples(args.data_dir)
    class_labels =Processor().get_labels()
    scriptsbase = "SuperGLUE/BoolQ"
    scriptformat = "txt"
    max_seq_l = 480 # this should be specified according to the running GPU's capacity
    if args.tune_plm: # tune the entire plm will use more gpu-memories, thus we should use a smaller batch_size.
        batchsize_t = 4
        batchsize_e = 4
        gradient_accumulation_steps = 8
        model_parallelize = True # if multiple gpus are available, one can use model_parallelize
    else:
        batchsize_t = 8
        batchsize_e = 4
        gradient_accumulation_steps = 4
        model_parallelize = False
elif args.dataset == "multirc":
    Processor = PROCESSORS["super_glue.multirc"]
    dataset['train'] = Processor().get_train_examples(args.data_dir)
    dataset['validation'] = Processor().get_dev_examples(args.data_dir)
    dataset['test'] = Processor().get_test_examples(args.data_dir)
    class_labels =Processor().get_labels()
    scriptsbase = "SuperGLUE/MultiRC"
    scriptformat = "txt"
    max_seq_l = 480
    if args.tune_plm:
        batchsize_t = 4
        batchsize_e = 4
        gradient_accumulation_steps = 8
        model_parallelize = True
    else:
        batchsize_t = 8
        batchsize_e = 4
        gradient_accumulation_steps = 4
        model_parallelize = False
elif args.dataset == "rte":
    Processor = PROCESSORS["super_glue.rte"]
    dataset['train'] = Processor().get_train_examples(args.data_dir)
    dataset['validation'] = Processor().get_dev_examples(args.data_dir)
    dataset['test'] = Processor().get_test_examples(args.data_dir)
    class_labels =Processor().get_labels()
    scriptsbase = "SuperGLUE/RTE"
    scriptformat = "txt"
    max_seq_l = 480
    if args.tune_plm:
        batchsize_t = 4
        batchsize_e = 4
        gradient_accumulation_steps = 2
        model_parallelize = True
    else:
        batchsize_t = 8
        batchsize_e = 4
        gradient_accumulation_steps = 4
        model_parallelize = False
elif args.dataset == "cb":
    Processor = PROCESSORS["super_glue.cb"]
    dataset['train'] = Processor().get_train_examples(args.data_dir)
    dataset['validation'] = Processor().get_dev_examples(args.data_dir)
    dataset['test'] = Processor().get_test_examples(args.data_dir)
    class_labels =Processor().get_labels()
    scriptsbase = "SuperGLUE/CB"
    scriptformat = "txt"
    max_seq_l = 480
    if args.tune_plm:
        batchsize_t = 4
        batchsize_e = 4
        gradient_accumulation_steps = 8
        model_parallelize = True
    else:
        batchsize_t = 8
        batchsize_e = 4
        gradient_accumulation_steps = 4
        model_parallelize = False
elif args.dataset == "wic":
    Processor = PROCESSORS["super_glue.wic"]
    dataset['train'] = Processor().get_train_examples(args.data_dir)
    dataset['validation'] = Processor().get_dev_examples(args.data_dir)
    dataset['test'] = Processor().get_test_examples(args.data_dir)
    class_labels =Processor().get_labels()
    scriptsbase = "SuperGLUE/WiC"
    scriptformat = "txt"
    max_seq_l = 480
    if args.tune_plm:
        batchsize_t = 4
        batchsize_e = 4
        gradient_accumulation_steps = 8
        model_parallelize = True
    else:
        batchsize_t = 8
        batchsize_e = 4
        gradient_accumulation_steps = 4
        model_parallelize = False
elif args.dataset == "fewshot_boolq":
    Processor = PROCESSORS["super_glue.boolq"]
    dataset['train'] = Processor().get_train_examples(args.data_dir)
    dataset['validation'] = Processor().get_dev_examples(args.data_dir)
    dataset['test'] = Processor().get_test_examples(args.data_dir)
    class_labels =Processor().get_labels()
    scriptsbase = "SuperGLUE/BoolQ"
    scriptformat = "txt"
    sampler = FewShotSampler(num_examples_per_label=args.shot)
    dataset['train']= sampler(dataset['train'], seed=args.seed)
    max_seq_l = 480
    if args.tune_plm:
        batchsize_t = 4
        batchsize_e = 4
        gradient_accumulation_steps = 8
        model_parallelize = True
    else:
        batchsize_t = 8
        batchsize_e = 4
        gradient_accumulation_steps = 4
        model_parallelize = False
elif args.dataset == "fewshot_multirc":
    Processor = PROCESSORS["super_glue.multirc"]
    dataset['train'] = Processor().get_train_examples(args.data_dir)
    dataset['validation'] = Processor().get_dev_examples(args.data_dir)
    dataset['test'] = Processor().get_test_examples(args.data_dir)
    class_labels =Processor().get_labels()
    scriptsbase = "SuperGLUE/MultiRC"
    scriptformat = "txt"
    sampler = FewShotSampler(num_examples_per_label=args.shot)
    dataset['train']= sampler(dataset['train'], seed=args.seed)
    max_seq_l = 480
    if args.tune_plm:
        batchsize_t = 4
        batchsize_e = 4
        gradient_accumulation_steps = 8
        model_parallelize = True
    else:
        batchsize_t = 8
        batchsize_e = 4
        gradient_accumulation_steps = 4
        model_parallelize = False
elif args.dataset == "fewshot_wic":
    Processor = PROCESSORS["super_glue.wic"]
    dataset['train'] = Processor().get_train_examples(args.data_dir)
    dataset['validation'] = Processor().get_dev_examples(args.data_dir)
    dataset['test'] = Processor().get_test_examples(args.data_dir)
    class_labels =Processor().get_labels()
    scriptsbase = "SuperGLUE/WiC"
    scriptformat = "txt"
    sampler = FewShotSampler(num_examples_per_label=args.shot)
    dataset['train']= sampler(dataset['train'], seed=args.seed)
    max_seq_l = 480
    if args.tune_plm:
        batchsize_t = 4
        batchsize_e = 4
        gradient_accumulation_steps = 8
        model_parallelize = True
    else:
        batchsize_t = 8
        batchsize_e = 4
        gradient_accumulation_steps = 4
        model_parallelize = False
elif args.dataset == "fewshot_cb":
    Processor = PROCESSORS["super_glue.cb"]
    dataset['train'] = Processor().get_train_examples(args.data_dir)
    dataset['validation'] = Processor().get_dev_examples(args.data_dir)
    dataset['test'] = Processor().get_test_examples(args.data_dir)
    class_labels =Processor().get_labels()
    scriptsbase = "SuperGLUE/CB"
    scriptformat = "txt"
    sampler = FewShotSampler(num_examples_per_label=args.shot)
    dataset['train']= sampler(dataset['train'], seed=args.seed)
    max_seq_l = 480
    if args.tune_plm:
        batchsize_t = 4
        batchsize_e = 4
        gradient_accumulation_steps = 8
        model_parallelize = True
    else:
        batchsize_t = 8
        batchsize_e = 4
        gradient_accumulation_steps = 4
        model_parallelize = False
elif args.dataset == "fewshot_rte":
    Processor = PROCESSORS["super_glue.rte"]
    dataset['train'] = Processor().get_train_examples(args.data_dir)
    dataset['validation'] = Processor().get_dev_examples(args.data_dir)
    dataset['test'] = Processor().get_test_examples(args.data_dir)
    class_labels =Processor().get_labels()
    scriptsbase = "SuperGLUE/RTE"
    scriptformat = "txt"
    sampler = FewShotSampler(num_examples_per_label=args.shot)
    dataset['train']= sampler(dataset['train'], seed=args.seed)
    max_seq_l = 480
    if args.tune_plm:
        batchsize_t = 4
        batchsize_e = 4
        gradient_accumulation_steps = 8
        model_parallelize = True
    else:
        batchsize_t = 8
        batchsize_e = 4
        gradient_accumulation_steps = 4
        model_parallelize = False
else:
    raise NotImplementedError


# Now define the template and verbalizer.
# Note that soft template can be combined with hard template, by loading the hard template from file.
# For example, the template in soft_template.txt is {}
# The choice_id 1 is the hard template
mytemplate = SoftTemplate_ins_pos(model=plm, tokenizer=tokenizer, num_tokens=args.soft_token_num, initialize_from_vocab=args.init_from_vocab).from_file(f"/nfs/users/weicheng/textCL/OpenPrompt/scripts/{scriptsbase}/soft_template.txt", choice=args.template_id)

myverbalizer = ManualVerbalizer(tokenizer, classes=class_labels).from_file(f"/nfs/users/weicheng/textCL/OpenPrompt/scripts/{scriptsbase}/manual_verbalizer.{scriptformat}", choice=args.verbalizer_id)
wrapped_example = mytemplate.wrap_one_example(dataset['train'][0])
#print(wrapped_example)


use_cuda = True
if args.multigpu:
    model_parallelize = True
else:
    model_parallelize = False
prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=(not args.tune_plm), plm_eval_mode=args.plm_eval_mode)
#prompt_model = torch.nn.DataParallel(prompt_model)

if use_cuda:
    prompt_model= prompt_model.cuda()

if model_parallelize:
    #device_map_ = infer_auto_device_map(prompt_model)
    prompt_model.parallelize()


train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, decoder_max_length=3,
    batch_size=batchsize_t,shuffle=True, teacher_forcing=False, predict_eos_token=False,
    truncate_method="tail")

validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, decoder_max_length=3,
    batch_size=batchsize_e,shuffle=False, teacher_forcing=False, predict_eos_token=False,
    truncate_method="tail")

# zero-shot test
test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, decoder_max_length=3,
    batch_size=batchsize_e,shuffle=False, teacher_forcing=False, predict_eos_token=False,
    truncate_method="tail")

print("truncate rate: {}".format(test_dataloader.tokenizer_wrapper.truncate_rate), flush=True)


# for name, para in prompt_model.template.named_parameters():
#     if para.requires_grad:
#         print(type(para.data),para.size(), para.device)

def evaluate(prompt_model, dataloader, desc):
    prompt_model.eval()
    allpreds = []
    alllabels = []

    for step, inputs in enumerate(dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
    return acc

from transformers import  AdamW, get_linear_schedule_with_warmup,get_constant_schedule_with_warmup  # use AdamW is a standard practice for transformer
from transformers.optimization import Adafactor, AdafactorSchedule  # use Adafactor is the default setting for T5
loss_func = torch.nn.CrossEntropyLoss()

tot_step = args.max_steps


if args.tune_plm: # normally we freeze the model when using soft_template. However, we keep the option to tune plm
    no_decay = ['bias', 'LayerNorm.weight'] # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if (not any(nd in n for nd in no_decay))], 'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=3e-5)
    scheduler1 = get_linear_schedule_with_warmup(
        optimizer1,
        num_warmup_steps=500, num_training_steps=tot_step)
else:
    optimizer1 = None
    scheduler1 = None


optimizer_grouped_parameters2 = [{'params': [p for name, p in prompt_model.template.named_parameters() if 'raw_embedding' not in name]}] # note that you have to remove the raw_embedding manually from the optimization
if args.optimizer.lower() == "adafactor":
    optimizer2 = Adafactor(optimizer_grouped_parameters2,
                            lr=args.prompt_lr,
                            relative_step=False,
                            scale_parameter=False,
                            warmup_init=False)  # when lr is 0.3, it is the same as the configuration of https://arxiv.org/abs/2104.08691
    scheduler2 = get_constant_schedule_with_warmup(optimizer2, num_warmup_steps=args.warmup_step_prompt) # when num_warmup_steps is 0, it is the same as the configuration of https://arxiv.org/abs/2104.08691
elif args.optimizer.lower() == "adamw":
    optimizer2 = AdamW(optimizer_grouped_parameters2, lr=args.prompt_lr) # usually lr = 0.5
    scheduler2 = get_linear_schedule_with_warmup(
                    optimizer2,
                    num_warmup_steps=args.warmup_step_prompt, num_training_steps=tot_step) # usually num_warmup_steps is 500




tot_loss = 0
log_loss = 0
best_val_acc = 0
glb_step = 0
actual_step = 0
leave_training = False

acc_traces = []
tot_train_time = 0
pbar_update_freq = 10
prompt_model.train()

temperature = 1.0
temp_min = 0.5

ANNEAL_RATE = 0.00003

writer = SummaryWriter('./log')
temp = temperature
pbar = tqdm(total=tot_step, desc="Train")
for epoch in range(1000000):
    print(f"Begin epoch {epoch}")

    for step, inputs in enumerate(train_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        tot_train_time -= time.time()
        logits = prompt_model(inputs)
        labels = inputs['label']
        loss = loss_func(logits, labels)
        loss.backward()
        tot_loss += loss.item()
        actual_step += 1


        if actual_step % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(prompt_model.parameters(), 1.0)
            glb_step += 1

            temp = np.maximum(temp * np.exp(-ANNEAL_RATE * glb_step / args.aneal), temp_min)
            global_var.set_value('temperature', temp)
            if (temp * 100) % 10 == 0:
                print("temperature: \n", temp)

            if glb_step % pbar_update_freq == 0:
                aveloss = (tot_loss - log_loss)/pbar_update_freq
                pbar.update(10)
                pbar.set_postfix({'loss': aveloss})
                log_loss = tot_loss


            if optimizer1 is not None:
                optimizer1.step()
                optimizer1.zero_grad()
            if scheduler1 is not None:
                scheduler1.step()
            if optimizer2 is not None:
                optimizer2.step()
                optimizer2.zero_grad()
            if scheduler2 is not None:
                scheduler2.step()

        tot_train_time += time.time()

        if actual_step % gradient_accumulation_steps == 0 and glb_step >0 and glb_step % args.eval_every_steps == 0:
            val_acc = evaluate(prompt_model, validation_dataloader, desc="Valid")
            writer.add_scalar('val_acc', val_acc, epoch)
            if val_acc >= best_val_acc:
                #torch.save(prompt_model.state_dict(),f"{args.project_root}{this_run_unicode}.ckpt")
                best_val_acc = val_acc

            acc_traces.append(val_acc)
            print("Glb_step {}, val_acc {}, average time {}".format(glb_step, val_acc, tot_train_time/actual_step ), flush=True)
            prompt_model.train()

        if glb_step > args.max_steps:
            leave_training = True
            break

    if leave_training:
        break


# # super_glue test split can not be evaluated without submitting the results to their website. So we skip it here and keep them as comments.
#
# prompt_model.load_state_dict(torch.load(f"{args.project_root}/ckpts/{this_run_unicode}.ckpt"))
# prompt_model = prompt_model.cuda()
# test_acc = evaluate(prompt_model, test_dataloader, desc="Test")
# test_acc = evaluate(prompt_model, test_dataloader, desc="Test")

# a simple measure for the convergence speed.
thres99 = 0.99*best_val_acc
thres98 = 0.98*best_val_acc
thres100 = best_val_acc
step100=step98=step99=args.max_steps
for val_time, acc in enumerate(acc_traces):
    if acc>=thres98:
        step98 = min(val_time*args.eval_every_steps, step98)
        if acc>=thres99:
            step99 = min(val_time*args.eval_every_steps, step99)
            if acc>=thres100:
                step100 = min(val_time*args.eval_every_steps, step100)


content_write += f"BestValAcc:{best_val_acc}\tEndValAcc:{acc_traces[-1]}\tcritical_steps:{[step98,step99,step100]}\n"
content_write += "\n"

print(content_write)

with open(f"{args.result_file}", "a") as fout:
    fout.write(content_write)

import os
os.remove(f"./{this_run_unicode}.ckpt")