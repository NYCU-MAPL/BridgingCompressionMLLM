# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyrightk notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from comet_ml import ExistingExperiment, Experiment
# from comet_ml import OfflineExperiment as Experiment
import tqdm
import argparse
import math
import random
import json
import sys
import os
import time
import logging
import datetime
import yaml

import torch
import torch.nn as nn
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision import transforms
from torchvision.utils import save_image

from compressai.zoo import image_models

from collections import OrderedDict
from dataloader import coco_Karpathy
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from llama import LLaMA, ModelArgs, Tokenizer, VisionModel
from llama import Transformer as llamaAdapter_Transformer

from adapter_model import Linear_Encoder

def setup_model_parallel():
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MP'] = '1'
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '222' + str(random.randint(0,9))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size

def load_llamaAdapter_v1(
    ckpt_path: str,
    param_path: str,
    tokenizer_path: str,
    instruct_adapter_path: str,
    caption_adapter_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    print("Loading")
    instruct_adapter_checkpoint = torch.load(
        instruct_adapter_path, map_location="cpu")
    caption_adapter_checkpoint = torch.load(
        caption_adapter_path, map_location="cpu")
    with open(param_path, "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    model_args.adapter_layer = int(
        instruct_adapter_checkpoint['adapter_query.weight'].shape[0] / model_args.adapter_len)
    model_args.cap_adapter_layer = int(
        caption_adapter_checkpoint['cap_adapter_query.weight'].shape[0] / model_args.cap_adapter_len)

    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = llamaAdapter_Transformer(model_args)

    # To reduce memory usuage
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt, strict=False)

    vision_model = VisionModel(model_args)

    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(instruct_adapter_checkpoint, strict=False)
    model.load_state_dict(caption_adapter_checkpoint, strict=False)
    vision_model.load_state_dict(caption_adapter_checkpoint, strict=False)

    generator = LLaMA(model, tokenizer, vision_model)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def psnr(self, output, target):
        mse = torch.mean((output - target) ** 2)
        if(mse == 0):
            return 100
        max_pixel = 1.
        psnr = 10 * torch.log10(max_pixel / mse)
        return torch.mean(psnr)

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )

        if output["x_hat"] != None:

            out["mse_loss"] = self.mse(torch.clamp(output["x_hat"],0,1), target)
            out["psnr"] = self.psnr(torch.clamp(output["x_hat"],0,1), target)

        return out

class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def CLIP_transform():
    return Compose([
            Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            CenterCrop(224),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

def Codec_transform():
    return Compose([
            Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            CenterCrop(256),
            ToTensor()
        ])

def Codec2CLIP_transform():
    return Compose([
            Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            CenterCrop(224),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])


def coco_caption_eval(args, results_file, split):
    filenames = {'val':'coco_karpathy_val_gt.json','test':'coco_karpathy_test_gt.json'}    

    annotation_file = os.path.join(args.dataset_path, "annotations", filenames[split])
    
    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')
    
    return coco_eval

def init(args, argv):
    
    if args.token_noise: 
        exp_name = args.exp_name + f"_{str(args.token_noise).replace('.','').zfill(3)}"
    else:
        exp_name = args.exp_name

    base_dir = f'{args.root}/{exp_name}/{args.quality_level}/'
    os.makedirs(base_dir, exist_ok=True)

    experiment = Experiment(
        api_key        = args.comet_apt_key,
        project_name   = args.comet_project_name,
        workspace      = args.comet_workspace,
    )

    base_dir = f'{args.root}/TEST_cap(coco_Karpathy)_{exp_name}/{args.quality_level}/'
    os.makedirs(base_dir, exist_ok=True)
    exp_name = f"TEST_cap(coco_Karpathy)_{exp_name}_{args.quality_level}"
        
    experiment.set_name(exp_name)
    Hyperparameters = vars(args)
    Hyperparameters["location"] = os.getlogin()
    experiment.log_parameters(Hyperparameters)

    return base_dir, experiment, exp_name

def setup_logger(log_dir):
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    log_file_handler = logging.FileHandler(log_dir, encoding='utf-8')
    log_file_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_file_handler)

    log_stream_handler = logging.StreamHandler(sys.stdout)
    log_stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_stream_handler)

    logging.info('Logging file is %s' % log_dir)

def test_cap_epoch(epoch, test_dataloader, codec, adapter, llama_adapter, criterion_rd, comet_experiment, stage='test', device = None, args = None):

    codec.eval()
    adapter.eval()
    adapter = adapter.to(device)

    all_generated_captions = []

    prompt = "Generate caption of this image :"
    
    bpp_loss     = AverageMeter()
    psnr_avg     = AverageMeter()


    with torch.no_grad():
        tqdm_meter = tqdm.tqdm(enumerate(test_dataloader), total=len(test_dataloader), leave=False, ascii=True)
        for i, (images, caption, cocoid) in tqdm_meter:

            B, C, H, W = images.shape
            images = images.to(device)

            prompts = [prompt for _ in range(B)]

            out_net = codec(images)
            out_criterion = criterion_rd(out_net, images)
            bpp_loss.update(out_criterion["bpp_loss"], images.shape[0])
            psnr_avg.update(out_criterion["psnr"], images.shape[0])

            feature = adapter(out_net['y_hat'])
            batch_generated_captions = llama_adapter.generate(imgs = feature, prompts = prompts, start_layer = 2)


            for j in range(B):
                if j == 0:
                    print(f"\n\ncocoid: {cocoid[j].item()}\nGen caption: {batch_generated_captions[j]}\nGT caption: {caption[j]}")
                all_generated_captions.append({"image_id": cocoid[j].item(), "caption": batch_generated_captions[j]})


            update_txt=f'[{i*len(images)}/{len(test_dataloader.dataset)}'
            tqdm_meter.set_postfix_str(update_txt, refresh=True)
            

    all_generated_captions_path = os.path.join(args.root, comet_experiment.get_name()[:-2], str(args.quality_level), "test_captions_res.json")
    with open(all_generated_captions_path, "w") as outfile: 
        print(f"Save results to {all_generated_captions_path}")
        json.dump(all_generated_captions, outfile)

    comet_experiment.log_code(all_generated_captions_path)
    coco_test = coco_caption_eval(args, all_generated_captions_path, 'test')
                       
    log_stats = {f'test_{k}': v for k, v in coco_test.eval.items()}
    with open(os.path.join(args.root, comet_experiment.get_name()[:-2], str(args.quality_level), "test_evaluate.txt"),"w") as f:
        f.write(json.dumps(log_stats) + "\n") 



    B1, B2, B3, B4 = log_stats['test_Bleu_1'], log_stats['test_Bleu_2'], log_stats['test_Bleu_3'], log_stats['test_Bleu_4']
    Cider_score    = log_stats['test_CIDEr']
    Meteor_score   = log_stats['test_METEOR']
    Rouge_score    = log_stats['test_ROUGE_L']


    print(f"Bleu1: {B1}\nBleu2: {B2}\nBleu3: {B3}\nBleu4: {B4}\nCider: {Cider_score}\nMeteor: {Meteor_score}\nRouge: {Rouge_score}")
          

    log = {
        f'{stage}/Bleu1'   : B1,
        f'{stage}/Bleu2'   : B2,
        f'{stage}/Bleu3'   : B3,
        f'{stage}/Bleu4'   : B4,
        f'{stage}/Cider'   : Cider_score,
        f'{stage}/Meteor'  : Meteor_score,
        f'{stage}/Rouge'   : Rouge_score,
        f'{stage}/bpp'     : bpp_loss.avg,
        f'{stage}/psnr'    : psnr_avg.avg,
    }

    comet_experiment.log_metrics(log)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-c",
        "--config",
        help="Path to config file",
    )
    parser.add_argument(
        '--name', 
        default=datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S'), 
        type=str,
        help='Result dir name', 
    )
    parser.add_argument(
        '--token_noise', 
        type=float,
    )
    
    given_configs, remaining = parser.parse_known_args(argv)
    with open(given_configs.config) as file:
        yaml_data= yaml.safe_load(file)
        parser.set_defaults(**yaml_data)

    parser.add_argument(
        "--exp_name",
        type=str
    )
    parser.add_argument(
        "--checkpoint",
        type=str
    )

    args = parser.parse_args(remaining)

    return args

def main(argv):
    args = parse_args(argv)


    base_dir, experiment, exp_name = init(args, argv)
    experiment.add_tags([args.location])

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    if not args.num_workers:
        args.num_workers = len(os.sched_getaffinity(0)) - 2 
    experiment.log_parameters({"number of workers": args.num_workers})

    setup_logger(base_dir + '/' + time.strftime('%Y%m%d_%H%M%S') + '.log')
    msg = f'======================= {args.exp_name} ======================='
    logging.info(msg)
    for k in args.__dict__:
        logging.info(k + ':' + str(args.__dict__[k]))
    logging.info('=' * len(msg))
    experiment.log_code(argv[-1])


    gpu_memory = (os.popen('nvidia-smi --query-gpu memory.total --format=csv').read()).split('\n')[1].split(' ')[0]
    print(f"Current GPU memory: {gpu_memory}")
    print(f"Number of workers: {args.num_workers}")

    device = f"cuda:{args.device}" if args.cuda and torch.cuda.is_available() else "cpu"
            
    trans = Codec_transform()
    test_dataset = coco_Karpathy(base_image_dir=args.dataset_path, labels_paths=args.labels_paths, split = 'test', transform = trans)
    print(f'Testing with {len(test_dataset)} examples using coco_Karpathy_split.')

    test_dataloader = torch.utils.data.DataLoader(
                        test_dataset, batch_size=args.test_batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=False
                    )


    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    llama_adapter = load_llamaAdapter_v1(
        args.llama_adapter_v1['ckpt_path'], args.llama_adapter_v1['param_path'], args.llama_adapter_v1['tokenizer_path'], args.llama_adapter_v1['instruct_adapter_path'], args.llama_adapter_v1['caption_adapter_path'], local_rank, world_size, 512, args.test_batch_size
    )

    criterion_rd = RateDistortionLoss()
    net = None
    adapter = None
    ql = args.quality_level
    net = image_models[args.model](quality=ql)
    net = net.to(device)
            
    in_features = 320
    adapter = Linear_Encoder(in_features=in_features, out_features=512, num_tokens=1, args=args)

    print("Loading "+str(args.checkpoint))
    ckpt = torch.load(args.checkpoint, map_location=device)

    new_state_dict = OrderedDict()
    checkpoint    = OrderedDict()

    if "state_dict" not in ckpt.keys():
        checkpoint["state_dict"] = ckpt
    else:
        checkpoint = ckpt

    for k, v in checkpoint["state_dict"].items():
        name = k
        if 'module.' in k:
            name = k.replace("module.", "")
        new_state_dict[name] = v

    new_state = {k: p for k, p in new_state_dict.items()}
    for k, p in net.named_parameters():
        if (k not in new_state):
            print(f"No weight: {k}")
            continue
        if new_state[k].shape != p.shape:
            print(f"Size mismatch: {k}")

    net.load_state_dict(new_state_dict, strict = True)
    adapter.load_state_dict(ckpt['adapter'], strict = True)

    
    test_cap_epoch(-1, test_dataloader, net, adapter, llama_adapter, criterion_rd, experiment, 'test', device = device, args = args)


if __name__ == "__main__":
    main(sys.argv[1:])