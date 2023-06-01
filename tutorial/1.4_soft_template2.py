from tqdm import tqdm
#from openprompt.data_utils import PROCESSORS
import os, sys
import torch
sys.path.append(os.path.abspath('/nfs/users/weicheng/textCL/OpenPrompt/openprompt'))
from huggingface_dataset2 import PROCESSORS
from openprompt.data_utils.huggingface_dataset import SuperglueMultiRCProcessor, SuperglueBoolQProcessor, SuperglueCBProcessor, SuperglueCOPAProcessor, SuperglueRTEProcessor, SuperglueWiCProcessor, SuperglueWSCProcessor, SuperglueRecordProcessor, YahooAnswersTopicsProcessor

from openprompt.data_utils.utils import InputExample
import argparse
import numpy as np
from datasets import load_dataset, load_metric

sys.path.append('/nfs/users/weicheng/textCL/OpenPrompt/openprompt')
import pipeline_base

# use lm-adapted version or t5-v1.1 checkpoint. Note that the original t5 checkpoint has been pretrained
# on part of GLUE dataset, thus should not be used.
from pipeline_base import PromptDataLoader
#from openprompt import PromptDataLoader
from openprompt.prompts import ManualVerbalizer
from openprompt.prompts import SoftTemplate
from openprompt import PromptForClassification
import time
import os
#
# import os
# os.environ['LD_LIBRARY_PATH'] = "/usr/local/cuda/lib64/"
import logging

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser("")
parser.add_argument("--shot", type=int, default=32)
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--seed", type=int, default=144)
parser.add_argument("--plm_eval_mode", action="store_true", help="whether to turn off the dropout in the freezed model. Set to true to turn off.")
parser.add_argument("--tune_plm", type=bool, default=False)
parser.add_argument("--multigpu", type=bool, default=False)
parser.add_argument("--model", type=str, default='mt5-lm', help="We test both t5 and t5-lm in this scripts, the corresponding tokenizerwrapper will be automatically loaded.")
parser.add_argument("--model_name_or_path", default='/nfs/users/weicheng/textCL/OpenPrompt/plm/mt5-small')#t5-large-lm-adapt/')
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



sys.path.append('/nfs/users/weicheng/textCL/OpenPrompt/openprompt/plms')

sys.path.append(os.getcwd())
import seq2seq2

# use lm-adapted version or t5-v1.1 checkpoint. Note that the original t5 checkpoint has been pretrained
# on part of GLUE dataset, thus should not be used.
from seq2seq2 import T5TokenizerWrapper, T5LMTokenizerWrapper2
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration
from openprompt.data_utils.data_sampler import FewShotSampler

from plms import load_plm
#from openprompt.plms import load_plm


def load_data(data_args, tokenizer, batch_size, max_len):
    # Loading a dataset from your local files.
    # CSV/JSON training and evaluation files are needed.
    data_files = {"train": data_args.train_file,
                  "validation": data_args.validation_file}

    # Get the test dataset: you can provide your own CSV/JSON test file (see below)
    # when you use `do_predict` without specifying a GLUE benchmark task.

    if data_args.test_file is not None:
        train_extension = data_args.train_file.split(".")[-1]
        test_extension = data_args.test_file.split(".")[-1]
        assert (
            test_extension == train_extension
        ), "`test_file` should have the same extension (csv or json) as `train_file`."
        data_files["test"] = data_args.test_file
    else:
        raise ValueError(
            "Need either a GLUE task or a test file for `do_predict`.")

    for key in data_files.keys():
        logger.info(f"load a local file for {key}: {data_files[key]}")

    if data_args.train_file.endswith(".csv"):
        # Loading a dataset from local csv files
        datasets = load_dataset("csv", data_files=data_files)
    else:
        # Loading a dataset from local json files
        datasets = load_dataset("json", data_files=data_files)
        # Labels
    # Trying to have good defaults here, don't hesitate to tweak to your needs.
    is_regression = datasets["train"].features["label"].dtype in [
        "float32", "float64"]
    if is_regression:
        num_labels = 1
    else:
        # A useful fast method:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
        label_list = datasets["train"].unique("label")
        label_list.sort()  # Let's sort it for determinism
        num_labels = len(label_list)
    train_dataset = datasets["train"]
    eval_dataset = datasets["validation"]
    predict_dataset = datasets["test"]

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
    non_label_column_names = [
        name for name in datasets["train"].column_names if name != "label"]
    if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
        sentence1_key, sentence2_key = "sentence1", "sentence2"
    else:
        if len(non_label_column_names) >= 2:
            sentence1_key, sentence2_key = non_label_column_names[:2]
        else:
            sentence1_key, sentence2_key = non_label_column_names[0], None

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (
                examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding="max_length",
                           max_length=max_len, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1)
                               for l in examples["label"]]
        return result

    datasets = datasets.map(
        preprocess_function,
        batched=True,
        batch_size=batch_size,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )
    train_iter, valid_iter, test_iter = datasets['train'], datasets['validation'], datasets['test']
    train_iter.set_format(type='torch', columns=['input_ids', 'label'])
    valid_iter.set_format(type='torch', columns=['input_ids', 'label'])
    test_iter.set_format(type='torch', columns=['input_ids', 'label'])

    train_iter = torch.utils.data.DataLoader(
        train_iter, batch_size=batch_size, shuffle=True)
    valid_iter = torch.utils.data.DataLoader(
        valid_iter, batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(
        test_iter, batch_size=batch_size, shuffle=True)
   
    tr_data_label_pairs = list(zip(train_dataset['sequence'], train_dataset['label']))
    random.shuffle(tr_data_label_pairs)
    dev_data_label_pairs = list(zip(eval_dataset['sequence'], eval_dataset['label']))
    random.shuffle(dev_data_label_pairs)
    te_data_label_pairs = list(zip(predict_dataset['sequence'], predict_dataset['label']))
    random.shuffle(te_data_label_pairs)
    

    return train_iter, valid_iter, test_iter, tr_data_label_pairs, dev_data_label_pairs, te_data_label_pairs


plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model_name_or_path)
dataset = {}


if args.dataset == "am_pr":
    batch_size = 8
    _, _, _, dataset['train'], dataset['validation'], dataset['test'] = load_data(
        args, tokenizer, batch_size)
elif args.dataset == "am_re":
    batch_size = 8
    _, _, _, dataset['train'], dataset['validation'], dataset['test'] = load_data(
        args, tokenizer, batch_size)
elif args.dataset == "se_im":
    batch_size = 8
    parser.add_argument("--train_file", type=str,
                        default='/nfs/service_data/aiad/datasets-for-evaluation/securities-report_covid-impact/SR_covid-impact_train.csv')
    parser.add_argument("--validation_file", type=str,
                        default='/nfs/service_data/aiad/datasets-for-evaluation/securities-report_covid-impact/SR_covid-impact_eval.csv')
    parser.add_argument("--test_file", type=str,
                        default='/nfs/service_data/aiad/datasets-for-evaluation/securities-report_covid-impact/SR_covid-impact_eval.csv')
    args = parser.parse_args()
    max_seq_l = 128#480  # this should be specified according to the running GPU's capacity
    _, _, _, dataset_tr, dataset_va, dataset_te = load_data(
        args, tokenizer, batch_size, max_seq_l)
    Processor = PROCESSORS["aiad.se_im"]
    proce = Processor()
    dataset['train'] = proce.get_examples(dataset_tr, "train")
    dataset['validation'] = proce.get_examples(dataset_va, "dev")
    dataset['test'] = proce.get_examples(dataset_te, "test")
    class_labels = proce.labels

    #class_labels = ["negative", "positive"]
    scriptsbase = "AIAD/SEIM"
    scriptformat = "txt"
    
    max_seq_l = 128#480  # this should be specified according to the running GPU's capacity
    # tune the entire plm will use more gpu-memories, thus we should use a smaller batch_size.
    if args.tune_plm:
        batchsize_t = 4
        batchsize_e = 4
        gradient_accumulation_steps = 8
        # if multiple gpus are available, one can use model_parallelize
        model_parallelize = True
    else:
        batchsize_t = 8
        batchsize_e = 4
        gradient_accumulation_steps = 4
        model_parallelize = False
elif args.dataset == "se_ra":
    batch_size = 8
    _, _, _, dataset['train'], dataset['validation'], dataset['test'] = load_data(
        args, tokenizer, batch_size)
elif args.dataset == "wo_ac":
    batch_size = 8
    _, _, _, dataset['train'], dataset['validation'], dataset['test'] = load_data(
        args, tokenizer, batch_size)


# Below are multiple dataset examples, including few-shot ones.
elif args.dataset == "boolq":
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
mytemplate = SoftTemplate(model=plm, tokenizer=tokenizer, num_tokens=args.soft_token_num, initialize_from_vocab=args.init_from_vocab).from_file(f"/nfs/users/weicheng/textCL/OpenPrompt/scripts/{scriptsbase}/soft_template.txt", choice=args.template_id)
myverbalizer = ManualVerbalizer(tokenizer, classes=class_labels).from_file(f"/nfs/users/weicheng/textCL/OpenPrompt/scripts/{scriptsbase}/manual_verbalizer.{scriptformat}", choice=args.verbalizer_id)
# wrapped_example = mytemplate.wrap_one_example(dataset['train'][0])
#print(wrapped_example)


use_cuda = True
if args.multigpu:
    model_parallelize = True
else:
    model_parallelize = False
prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=(not args.tune_plm), plm_eval_mode=args.plm_eval_mode)
if use_cuda:
    prompt_model=  prompt_model.cuda()

if model_parallelize:
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
            if val_acc >= best_val_acc:
                # torch.save(prompt_model.state_dict(),f"{args.project_root}{this_run_unicode}.ckpt")
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
os.remove(f"../ckpts/{this_run_unicode}.ckpt")