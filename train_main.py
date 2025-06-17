import json
import logging
import os
import random
import sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import torch
from tqdm import tqdm
from config import get_params
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, PrefixTuningTemplate
from openprompt import PromptDataLoader, PromptForGeneration
from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from evaluate import universal_evaluate_snips
from reader import snips_data_reader

args = get_params()

domain2slot = {
    "AddToPlaylist": ['music_item', 'playlist_owner', 'entity_name', 'playlist', 'artist'],
    "BookRestaurant": ['city', 'facility', 'timeRange', 'restaurant_name', 'country', 'cuisine', 'restaurant_type', 'served_dish', 'party_size_number', 'poi', 'sort', 'spatial_relation', 'state', 'party_size_description'],
    "GetWeather": ['city', 'state', 'timeRange', 'current_location', 'country', 'spatial_relation', 'geographic_poi', 'condition_temperature', 'condition_description'],
    "PlayMusic": ['genre', 'music_item', 'service', 'year', 'playlist', 'album','sort', 'track', 'artist'],
    "RateBook": ['object_part_of_series_type', 'object_select', 'rating_value', 'object_name', 'object_type', 'rating_unit', 'best_rating'],
    "SearchCreativeWork": ['object_name', 'object_type'],
    "SearchScreeningEvent": ['timeRange', 'movie_type', 'object_location_type','object_type', 'location_name', 'spatial_relation', 'movie_name']
}

# set random seed
if args.random_seed >= 0:
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

# model save path
model_saved_dir = os.path.join(args.model_saved_path,args.model_name)
if not os.path.exists(model_saved_dir):
    os.makedirs(model_saved_dir)
params_json = args.__dict__
with open(os.path.join(model_saved_dir,'params.json'), 'w')as fout:
    json.dump(params_json, fout,indent=4)

# log info
log = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stderr, level=logging.INFO)
fh = logging.FileHandler(os.path.join(model_saved_dir,"log.txt"))
log.addHandler(fh)

# dataset
dataset = snips_data_reader()

# get plm and plm_config
plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model_name_or_path)

mytemplate = PrefixTuningTemplate(
        model=plm,
        tokenizer=tokenizer,
        text='{"placeholder":"text_a"}. {"placeholder":"text_b"} : {"special": "<eos>"}{"mask"}',
        prefix_dropout=args.prefix_dp,
        using_decoder_past_key_values=True,
        num_token=args.prefix_num_tokens,
        )

# target domain slots number
slot_nums = len(domain2slot[args.target_domain])

# load dataset
if not args.test_only:
    train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer, 
        tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=256, 
        batch_size=args.batch_size, shuffle=True, teacher_forcing=True, predict_eos_token=True,
        truncate_method="head")

    validation_dataloader = PromptDataLoader(dataset=dataset["val"], template=mytemplate, tokenizer=tokenizer, 
        tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=256, 
        batch_size=slot_nums, shuffle=False, teacher_forcing=False, predict_eos_token=True,
        truncate_method="head")

test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer, 
    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=256, 
    batch_size=slot_nums, shuffle=False, teacher_forcing=False, predict_eos_token=True,
    truncate_method="head")

# parameters optimizer
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
{
    "params": [p for n, p in mytemplate.named_parameters() if (not any(nd in n for nd in no_decay)) and p.requires_grad],
    "weight_decay": 0.0,
},
{
    "params": [p for n, p in mytemplate.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
    "weight_decay": 0.0,
},
]

optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)
if not args.test_only:
    tot_step = (len(train_dataloader) // args.batch_size) * args.max_epochs if len(train_dataloader) % args.batch_size == 0 else (len(train_dataloader) // args.batch_size + 1) * args.max_epochs 
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, tot_step)

# test
if args.test_only:
    template_model = torch.load(os.path.join(model_saved_dir,"best_model.pth"))
    mytemplate.load_state_dict(template_model['template']) 
    prompt_model = PromptForGeneration(plm=plm,template=mytemplate,tokenizer=tokenizer)
    if args.use_cuda:
        prompt_model = prompt_model.cuda()

    log.info("evaluating begin")
    
    res = universal_evaluate_snips(prompt_model,test_dataloader)
    print(f"best result is {res}")
    print("-------------------------------------------------------")

# train 
else:
    prompt_model = PromptForGeneration(plm=plm,template=mytemplate, freeze_plm=True,tokenizer=tokenizer, plm_eval_mode=args.plm_eval_mode)
    if args.use_cuda:
        prompt_model.cuda()
    
    log_loss = 0
    tot_loss = 0
    best_dev_acc = 0.0
    cur_dev_acc = 0.0
    stop_steps_count = 0
    stop_flags = False

    for epoch in range(args.max_epochs):
        pbar = tqdm(enumerate(train_dataloader),total=len(train_dataloader))
        log.info("-----------------------------------------")
        log.info("training begining")
        log.info("-----------------------------------------")
        prompt_model.train()
        global_step = 0
        if stop_flags:
            break
        else:
            for step, inputs in pbar:
                global_step +=1
                if args.use_cuda:
                    input_sentence = inputs.cuda()

                loss = prompt_model(input_sentence)
                loss.backward()

                tot_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(mytemplate.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                pbar.set_description(f"Epoch {epoch}, global_step {global_step}, loss: {'%.2f'%loss.item()}")
                if global_step % 500 == 0 or global_step == len(train_dataloader) - 1: 
                    print("Epoch {}, global_step {} average loss: {} lr: {}".format(epoch, global_step, (tot_loss-log_loss)/ 500, scheduler.get_last_lr()[0]), flush=True)
                    log_loss = tot_loss
                    
                    # save best model
                    cur_dev_acc = universal_evaluate_snips(prompt_model,validation_dataloader)
                    log.info(f"cur_dev_acc is {cur_dev_acc}")
                    if cur_dev_acc > best_dev_acc:
                        best_dev_acc = cur_dev_acc
                        log.info(f"found better model")
                        torch.save(prompt_model.state_dict(),os.path.join(model_saved_dir,"best_model.pth"))
                        cur_test_acc = universal_evaluate_snips(prompt_model,test_dataloader)
                        log.info("-----------------------------------------")
                        log.info(f"best result is {cur_test_acc}")
                        log.info("-----------------------------------------")
                        stop_steps_count = 0
                    
                    else:
                        stop_steps_count += 1
                        log.info(f"No better model found {stop_steps_count}/{args.early_stop}")
                        if stop_steps_count == args.early_stop:
                            stop_flags = True
                            log.info(f"training finished")
                            break
