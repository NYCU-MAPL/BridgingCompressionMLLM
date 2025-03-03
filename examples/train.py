from comet_ml import ExistingExperiment, Experiment
import warnings
import tqdm
import argparse
import math
import random
import shutil
import sys
import os
import time
import logging
import gc
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, InterpolationMode

from compressai.zoo import image_models
from collections import OrderedDict

import clip
import CLIP_modify.clip as clip_modify

from adapter_model import Linear_Encoder
warnings.filterwarnings("ignore")


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

class ClipClsloss(nn.Module):
    def __init__(self, args, device) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.args = args
        self.device = device
        self.clip_model, _ = clip_modify.load("ViT-L/14", device=device)
        self.clip_model.eval()
        self.zeroshot_weights = self.zeroshot_classifier() # [512, 1000]
        self.processing = self._transform(224)

    def forward(self, output, labels, original_image = None, saved_token_count = None, exp_name = None):

        
        output, _ = self.clip_model.encode_image(output, start_layer = 2) 
        logits = output.type(self.zeroshot_weights.dtype) @ self.zeroshot_weights 
        assert logits.shape == (output.shape[0], 1000)
        loss = {}
        loss['clipcls'] = self.ce(logits, labels)


        #### Predect feature ####
        pred_feat = []
        pred_feat.append(output.clone())
        

        #### Original feature ####
        image = self.processing(original_image)
        with torch.no_grad():
            output, _ = self.clip_model.encode_image(image, start_layer = 0) 

            ori_feat = []
            ori_feat.append(output.detach().clone())

        perc_loss = torch.stack([nn.functional.mse_loss(p,o, reduction='none') for p,o in zip(pred_feat, ori_feat)]).squeeze()


        loss['clip_distill'] =  perc_loss.mean()

        accu = {}
        if labels is not None:
            top1, top5 = self.accuracy(logits, labels, topk=(1, 5))

            accu['top1'] = top1/logits.shape[0]
            accu['top5'] = top5/logits.shape[0]

        return loss, accu

    def _transform(self, n_px):
        return Compose([
            Resize(n_px, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(n_px),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def zeroshot_classifier(self):
        classnames = [
            "tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray", "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco", "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee", "American dipper", "kite (bird of prey)", "bald eagle", "vulture", "great grey owl", "fire salamander", "smooth newt", "newt", "spotted salamander", "axolotl", "American bullfrog", "tree frog", "tailed frog", "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin", "box turtle", "banded gecko", "green iguana", "Carolina anole", "desert grassland whiptail lizard", "agama", "frilled-necked lizard", "alligator lizard", "Gila monster", "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile", "American alligator", "triceratops", "worm snake", "ring-necked snake", "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake", "water snake", "vine snake", "night snake", "boa constrictor", "African rock python", "Indian cobra", "green mamba", "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake", "sidewinder rattlesnake", "trilobite", "harvestman", "scorpion", "yellow garden spider", "barn spider", "European garden spider", "southern black widow", "tarantula", "wolf spider", "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peafowl", "quail", "partridge", "african grey parrot", "macaw", "sulphur-crested cockatoo", "lorikeet", "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "duck", "red-breasted merganser", "goose", "black swan", "tusker", "echidna", "platypus", "wallaby", "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode", "conch", "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab", "fiddler crab", "red king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab", "isopod", "white stork", "black stork", "spoonbill", "flamingo", "little blue heron", "great egret", "bittern bird", "crane bird", "limpkin", "common gallinule", "American coot", "bustard", "ruddy turnstone", "dunlin", "common redshank", "dowitcher", "oystercatcher", "pelican", "king penguin", "albatross", "grey whale", "killer whale", "dugong", "sea lion", "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel", "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound", "Beagle", "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound", "Treeing Walker Coonhound", "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound", "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound", "Weimaraner", "Staffordshire Bull Terrier", "American Staffordshire Terrier", "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier", "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier", "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier", "Australian Terrier", "Dandie Dinmont Terrier", "Boston Terrier", "Miniature Schnauzer", "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier", "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier", "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever", "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla", "English Setter", "Irish Setter", "Gordon Setter", "Brittany dog", "Clumber Spaniel", "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel", "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael dog", "Malinois", "Briard", "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie", "Border Collie", "Bouvier des Flandres dog", "Rottweiler", "German Shepherd Dog", "Dobermann", "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog", "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff", "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute", "Siberian Husky", "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland dog", "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "brussels griffon", "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle", "Standard Poodle", "Mexican hairless dog (xoloitzcuintli)", "grey wolf", "Alaskan tundra wolf", "red wolf or maned wolf", "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox", "kit fox", "Arctic fox", "grey fox", "tabby cat", "tiger cat", "Persian cat", "Siamese cat", "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger", "cheetah", "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose", "meerkat", "tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle", "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper", "cricket insect", "stick insect", "cockroach", "praying mantis", "cicada", "leafhopper", "lacewing", "dragonfly", "damselfly", "red admiral butterfly", "ringlet butterfly", "monarch butterfly", "small white butterfly", "sulphur butterfly", "gossamer-winged butterfly", "starfish", "sea urchin", "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit", "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig", "common sorrel horse", "zebra", "pig", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison", "ram (adult male sheep)", "bighorn sheep", "Alpine ibex", "hartebeest", "impala (antelope)", "gazelle", "arabian camel", "llama", "weasel", "mink", "European polecat", "black-footed ferret", "otter", "skunk", "badger", "armadillo", "three-toed sloth", "orangutan", "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey", "baboon", "macaque", "langur", "black-and-white colobus", "proboscis monkey", "marmoset", "white-headed capuchin", "howler monkey", "titi monkey", "Geoffroy's spider monkey", "common squirrel monkey", "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant", "red panda", "giant panda", "snoek fish", "eel", "silver salmon", "rock beauty fish", "clownfish", "sturgeon", "gar fish", "lionfish", "pufferfish", "abacus", "abaya", "academic gown", "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance", "amphibious vehicle", "analog clock", "apiary", "apron", "trash can", "assault rifle", "backpack", "bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid", "banjo", "baluster / handrail", "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel", "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon", "swimming cap", "bath towel", "bathtub", "station wagon", "lighthouse", "beaker", "military hat (bearskin or shako)", "beer bottle", "beer glass", "bell tower", "baby bib", "tandem bicycle", "bikini", "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh", "bolo tie", "poke bonnet", "bookcase", "bookstore", "bottle cap", "hunting bow", "bow tie", "brass memorial plaque", "bra", "breakwater", "breastplate", "broom", "bucket", "buckle", "bulletproof vest", "high-speed train", "butcher shop", "taxicab", "cauldron", "candle", "cannon", "canoe", "can opener", "cardigan", "car mirror", "carousel", "tool kit", "cardboard box / carton", "car wheel", "automated teller machine", "cassette", "cassette player", "castle", "catamaran", "CD player", "cello", "mobile phone", "chain", "chain-link fence", "chain mail", "chainsaw", "storage chest", "chiffonier", "bell or wind chime", "china cabinet", "Christmas stocking", "church", "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs", "cocktail shaker", "coffee mug", "coffeemaker", "spiral or coil", "combination lock", "computer keyboard", "candy store", "container ship", "convertible", "corkscrew", "cornet", "cowboy boot", "cowboy hat", "cradle", "construction crane", "crash helmet", "crate", "infant bed", "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer", "rotary dial telephone", "diaper", "digital clock", "digital watch", "dining table", "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig", "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar", "electric locomotive", "entertainment center", "envelope", "espresso machine", "face powder", "feather boa", "filing cabinet", "fireboat", "fire truck", "fire screen", "flagpole", "flute", "folding chair", "football helmet", "forklift", "fountain", "fountain pen", "four-poster bed", "freight car", "French horn", "frying pan", "fur coat", "garbage truck", "gas mask or respirator", "gas pump", "goblet", "go-kart", "golf ball", "golf cart", "gondola", "gong", "gown", "grand piano", "greenhouse", "radiator grille", "grocery store", "guillotine", "hair clip", "hair spray", "half-track", "hammer", "hamper", "hair dryer", "hand-held computer", "handkerchief", "hard disk drive", "harmonica", "harp", "combine harvester", "hatchet", "holster", "home theater", "honeycomb", "hook", "hoop skirt", "gymnastic horizontal bar", "horse-drawn vehicle", "hourglass", "iPod", "clothes iron", "carved pumpkin", "jeans", "jeep", "T-shirt", "jigsaw puzzle", "rickshaw", "joystick", "kimono", "knee pad", "knot", "lab coat", "ladle", "lampshade", "laptop computer", "lawn mower", "lens cap", "letter opener", "library", "lifeboat", "lighter", "limousine", "ocean liner", "lipstick", "slip-on shoe", "lotion", "music speaker", "loupe magnifying glass", "sawmill", "magnetic compass", "messenger bag", "mailbox", "tights", "one-piece bathing suit", "manhole cover", "maraca", "marimba", "mask", "matchstick", "maypole", "maze", "measuring cup", "medicine cabinet", "megalith", "microphone", "microwave oven", "military uniform", "milk can", "minibus", "miniskirt", "minivan", "missile", "mitten", "mixing bowl", "mobile home", "ford model t", "modem", "monastery", "monitor", "moped", "mortar and pestle", "graduation cap", "mosque", "mosquito net", "vespa", "mountain bike", "tent", "computer mouse", "mousetrap", "moving van", "muzzle", "metal nail", "neck brace", "necklace", "baby pacifier", "notebook computer", "obelisk", "oboe", "ocarina", "odometer", "oil filter", "pipe organ", "oscilloscope", "overskirt", "bullock cart", "oxygen mask", "product packet / packaging", "paddle", "paddle wheel", "padlock", "paintbrush", "pajamas", "palace", "pan flute", "paper towel", "parachute", "parallel bars", "park bench", "parking meter", "railroad car", "patio", "payphone", "pedestal", "pencil case", "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum", "Pickelhaube", "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball", "pinwheel", "pirate ship", "drink pitcher", "block plane", "planetarium", "plastic bag", "plate rack", "farm plow", "plunger", "Polaroid camera", "pole", "police van", "poncho", "pool table", "soda bottle", "plant pot", "potter's wheel", "power drill", "prayer rug", "printer", "prison", "missile", "projector", "hockey puck", "punching bag", "purse", "quill", "quilt", "race car", "racket", "radiator", "radio", "radio telescope", "rain barrel", "recreational vehicle", "fishing casting reel", "reflex camera", "refrigerator", "remote control", "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "eraser", "rugby ball", "ruler measuring stick", "sneaker", "safe", "safety pin", "salt shaker", "sandal", "sarong", "saxophone", "scabbard", "weighing scale", "school bus", "schooner", "scoreboard", "CRT monitor", "screw", "screwdriver", "seat belt", "sewing machine", "shield", "shoe store", "shoji screen / room divider", "shopping basket", "shopping cart", "shovel", "shower cap", "shower curtain", "ski", "balaclava ski mask", "sleeping bag", "slide rule", "sliding door", "slot machine", "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock", "solar thermal collector", "sombrero", "soup bowl", "keyboard space bar", "space heater", "space shuttle", "spatula", "motorboat", "spider web", "spindle", "sports car", "spotlight", "stage", "steam locomotive", "through arch bridge", "steel drum", "stethoscope", "scarf", "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher", "couch", "stupa", "submarine", "suit", "sundial", "sunglasses", "sunglasses", "sunscreen", "suspension bridge", "mop", "sweatshirt", "swim trunks / shorts", "swing", "electrical switch", "syringe", "table lamp", "tank", "tape player", "teapot", "teddy bear", "television", "tennis ball", "thatched roof", "front curtain", "thimble", "threshing machine", "throne", "tile roof", "toaster", "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck", "toy store", "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle", "trimaran", "tripod", "triumphal arch", "trolleybus", "trombone", "hot tub", "turnstile", "typewriter keyboard", "umbrella", "unicycle", "upright piano", "vacuum cleaner", "vase", "vaulted or arched ceiling", "velvet fabric", "vending machine", "vestment", "viaduct", "violin", "volleyball", "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft", "sink", "washing machine", "water bottle", "water jug", "water tower", "whiskey jug", "whistle", "hair wig", "window screen", "window shade", "Windsor tie", "wine bottle", "airplane wing", "wok", "wooden spoon", "wool", "split-rail fence", "shipwreck", "sailboat", "yurt", "website", "comic book", "crossword", "traffic or street sign", "traffic light", "dust jacket", "menu", "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "popsicle", "baguette", "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potatoes", "cabbage", "broccoli", "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber", "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith apple", "strawberry", "orange", "lemon", "fig", "pineapple", "banana", "jackfruit", "cherimoya (custard apple)", "pomegranate", "hay", "carbonara", "chocolate syrup", "dough", "meatloaf", "pizza", "pot pie", "burrito", "red wine", "espresso", "tea cup", "eggnog", "mountain", "bubble", "cliff", "coral reef", "geyser", "lakeshore", "promontory", "sandbar", "beach", "valley", "volcano", "baseball player", "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn", "rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra", "stinkhorn mushroom", "earth star fungus", "hen of the woods mushroom", "bolete", "corn cob", "toilet paper"]
        templates  = [
            'a bad photo of a {}.',
            'a photo of many {}.',
            'a sculpture of a {}.',
            'a photo of the hard to see {}.',
            'a low resolution photo of the {}.',
            'a rendering of a {}.',
            'graffiti of a {}.',
            'a bad photo of the {}.',
            'a cropped photo of the {}.',
            'a tattoo of a {}.',
            'the embroidered {}.',
            'a photo of a hard to see {}.',
            'a bright photo of a {}.',
            'a photo of a clean {}.',
            'a photo of a dirty {}.',
            'a dark photo of the {}.',
            'a drawing of a {}.',
            'a photo of my {}.',
            'the plastic {}.',
            'a photo of the cool {}.',
            'a close-up photo of a {}.',
            'a black and white photo of the {}.',
            'a painting of the {}.',
            'a painting of a {}.',
            'a pixelated photo of the {}.',
            'a sculpture of the {}.',
            'a bright photo of the {}.',
            'a cropped photo of a {}.',
            'a plastic {}.',
            'a photo of the dirty {}.',
            'a jpeg corrupted photo of a {}.',
            'a blurry photo of the {}.',
            'a photo of the {}.',
            'a good photo of the {}.',
            'a rendering of the {}.',
            'a {} in a video game.',
            'a photo of one {}.',
            'a doodle of a {}.',
            'a close-up photo of the {}.',
            'a photo of a {}.',
            'the origami {}.',
            'the {} in a video game.',
            'a sketch of a {}.',
            'a doodle of the {}.',
            'a origami {}.',
            'a low resolution photo of a {}.',
            'the toy {}.',
            'a rendition of the {}.',
            'a photo of the clean {}.',
            'a photo of a large {}.',
            'a rendition of a {}.',
            'a photo of a nice {}.',
            'a photo of a weird {}.',
            'a blurry photo of a {}.',
            'a cartoon {}.',
            'art of a {}.',
            'a sketch of the {}.',
            'a embroidered {}.',
            'a pixelated photo of a {}.',
            'itap of the {}.',
            'a jpeg corrupted photo of the {}.',
            'a good photo of a {}.',
            'a plushie {}.',
            'a photo of the nice {}.',
            'a photo of the small {}.',
            'a photo of the weird {}.',
            'the cartoon {}.',
            'art of the {}.',
            'a drawing of the {}.',
            'a photo of the large {}.',
            'a black and white photo of a {}.',
            'the plushie {}.',
            'a dark photo of a {}.',
            'itap of a {}.',
            'graffiti of the {}.',
            'a toy {}.',
            'itap of my {}.',
            'a photo of a cool {}.',
            'a photo of a small {}.',
            'a tattoo of the {}.',
        ]

        os.makedirs('ImageNet_CLIP_TextEmb', exist_ok=True)
        path = f'ImageNet_CLIP_TextEmb/ImageNet_text_CLIP_emb_ViT-L-14.pt'
        if os.path.isfile(path):
            zeroshot_weights = torch.load(path)
            zeroshot_weights = zeroshot_weights.cuda()
        else:
            with torch.no_grad():
                zeroshot_weights = []
                for classname in tqdm.tqdm(classnames):
                    texts = [template.format(classname) for template in templates] #format with class
                    texts = clip.tokenize(texts).cuda() #tokenize
                    class_embeddings = self.clip_model.encode_text(texts) #embed with text encoder
                    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                    class_embedding = class_embeddings.mean(dim=0)
                    class_embedding /= class_embedding.norm()
                    zeroshot_weights.append(class_embedding)
                zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
                torch.save(zeroshot_weights, path)

        return zeroshot_weights

    def accuracy(self, output, target, topk=(1,)):
        pred = output.topk(max(topk), 1, True, True)[1].t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

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



def init(args, argv):
    
    base_dir = f'{args.root}/{args.exp_name}/{args.quality_level}/'
    os.makedirs(base_dir, exist_ok=True)

    if args.resume_exp_key is not None:
        experiment = ExistingExperiment(
            api_key        = args.comet_apt_key,
            project_name   = args.comet_project_name,
            experiment_key = args.resume_exp_key,
            workspace      = args.comet_workspace
        )
    else:
        experiment = Experiment(
            api_key        = args.comet_apt_key,
            project_name   = args.comet_project_name,
            workspace      = args.comet_workspace
    )

    exp_name = f"{args.exp_name}_{args.quality_level}"
        
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

def configure_optimizers(adapter, args):
    """Set optimizer for only the parameters for propmts"""


    adapter_params = {
        n
        for n, p in adapter.named_parameters()
        if p.requires_grad
    }
    print(f"Adapter parameters: {adapter_params}")

    adapter_params_dict = dict(adapter.named_parameters())


    params = list(adapter_params_dict[n] for n in adapter_params)
    optimizer = torch.optim.Adam(params, lr=args.optimizer['adapter']['learning_rate'])

    return optimizer, None


def train_clipcls_one_epoch(model, criterion_rd, criterion_clip_cls, train_dataloader, optimizer, aux_optimizer, epoch, comet_experiment, args = None, adapter = None):
    
    avg_total_loss = AverageMeter()
    avg_bpp_loss = AverageMeter()
    avg_mse_loss = AverageMeter()
    avg_clip_distill_loss = AverageMeter()
    avg_clip_cls_loss = AverageMeter()
    avg_accu_top1 = AverageMeter()
    avg_accu_top5 = AverageMeter()
    avg_psnr    = AverageMeter()

    model.train()
    adapter.train()
    device = next(model.parameters()).device


    tqdm_emu = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=False, ascii=True)
    step = len(train_dataloader) * epoch
    for i, (images, label) in tqdm_emu:
        comet_experiment.set_step(step)
        step += 1

        images = images.to(device)
        label  = label.to(device)

        optimizer.zero_grad()
        if aux_optimizer:
            aux_optimizer.zero_grad()
        out_net = model(images)
        clip_image_feature =  adapter(out_net['y_hat'])

        out_criterion            = criterion_rd(out_net , images)
        clip_criterion, accu = criterion_clip_cls(clip_image_feature, label, original_image = images)


        if epoch < 20:
            distortion = 100 * clip_criterion['clipcls']
            loss_list = ['clipcls']
        elif epoch >= 20 and epoch < 40:
            distortion = 1 * clip_criterion['clipcls'] + 100 * clip_criterion['clip_distill']
            loss_list = ['clipcls', 'clip_distill']
        else:
            distortion = 100 * clip_criterion['clip_distill']
            loss_list = ['clip_distill']


        total_loss = 11.52 * distortion
        update_txt=f'[{i*len(images)}/{len(train_dataloader.dataset)}] epoch: {epoch} | Loss: {total_loss.item():.3f} | Bpp loss: {out_criterion["bpp_loss"].item():.4f} |  top1: {avg_accu_top1.avg:.4f} | Loss list: {loss_list} | lr: {optimizer.param_groups[-1]["lr"]}'

        if "ComputeAllTok" not in args.exp_name:
            total_loss.backward()
            optimizer.step()

        if aux_optimizer:
            aux_loss = model.aux_loss()
            aux_loss.backward()
            aux_optimizer.step()


        tqdm_emu.set_postfix_str(update_txt, refresh=True)

        with torch.no_grad():
            avg_total_loss.update(total_loss.item(), n = len(images))
            avg_bpp_loss.update(out_criterion["bpp_loss"].item(), n = len(images))
            avg_mse_loss.update(out_criterion['mse_loss'].item(), n = len(images))
            avg_clip_cls_loss.update(clip_criterion['clipcls'].item(), n = len(images))
            avg_psnr.update(out_criterion['psnr'], n = len(images))
            avg_accu_top1.update(accu['top1'], n = len(images))
            avg_accu_top5.update(accu['top5'], n = len(images))
            avg_clip_distill_loss.update(clip_criterion['clip_distill'].item(), n = len(images))

        log = {
            'train/total_loss':         avg_total_loss.avg,
            'train/bpp loss':           avg_bpp_loss.avg,
            'train/mse loss':           avg_mse_loss.avg,
            'train/clip_cls loss':      avg_clip_cls_loss.avg,
            'train/psnr':               avg_psnr.avg,
            'train/accu top1':          avg_accu_top1.avg,
            'train/accu top5':          avg_accu_top5.avg,
            'train/clip_distill loss':  avg_clip_distill_loss.avg
        } 

        comet_experiment.log_metrics(log)

    torch.cuda.empty_cache()
    gc.collect()


def test_clipcls_epoch(epoch, test_dataloader, model, criterion_rd, criterion_clip_cls, comet_experiment, stage='test', args = None, sanity = False, adapter = None):
    model.eval()
    adapter.eval()
    device = next(model.parameters()).device

    bpp_loss      = AverageMeter()
    clip_cls_loss = AverageMeter()
    clip_distill_loss = AverageMeter()
    top1_avg     = AverageMeter()
    top5_avg     = AverageMeter()
    totalloss    = AverageMeter()
    psnr_avg     = AverageMeter() 

    count = 0 # for saving token

    with torch.no_grad():
        tqdm_meter = tqdm.tqdm(enumerate(test_dataloader),leave=False, total=len(test_dataloader), ascii=True)
        for i, (images, label) in tqdm_meter:

            images = images.to(device)
            label  = label.to(device)

            out_net = model(images)
            clip_image_feature =  adapter(out_net['y_hat'])
            
            out_criterion           = criterion_rd(out_net , images)
            clip_criterion, accu    = criterion_clip_cls(clip_image_feature, label, original_image = images, saved_token_count = count, exp_name = comet_experiment.get_name()[:-2],)


            if epoch < 20:
                distortion = 100 * clip_criterion['clipcls']
                loss_list = ['clipcls']
            elif epoch >= 20 and epoch < 40:
                distortion = 1 * clip_criterion['clipcls'] + 100 * clip_criterion['clip_distill']
                loss_list = ['clipcls', 'clip_distill']
            else:
                distortion = 100 * clip_criterion['clip_distill']
                loss_list = ['clip_distill']

            total_loss = 11.52 * distortion 

            bpp_loss.update(out_criterion["bpp_loss"], images.shape[0])
            psnr_avg.update(out_criterion["psnr"], images.shape[0])
            top1_avg.update(accu['top1'], images.shape[0])
            top5_avg.update(accu['top5'], images.shape[0])
            clip_cls_loss.update(clip_criterion['clipcls'], images.shape[0])
            totalloss.update(total_loss, images.shape[0])
            clip_distill_loss.update(clip_criterion['clip_distill'], images.shape[0])


            txt = f"epoch: {epoch} | {stage} | Total Loss: {totalloss.avg:.3f}  Bpp loss: {bpp_loss.avg:.4f} | top1: {top1_avg.avg:.3f} | Loss list: {loss_list}"
            tqdm_meter.set_postfix_str(txt)
            

            if sanity and i == 3:
                break

    log = {
        f'{stage}/bpp loss'          :bpp_loss.avg,
        f'{stage}/psnr'              :psnr_avg.avg,
        f'{stage}/clip_cls_loss loss':clip_cls_loss.avg,
        f"{stage}/total loss"        :totalloss.avg,
        f"{stage}/accu top1"         :top1_avg.avg,
        f"{stage}/accu top5"         :top5_avg.avg,
        f'{stage}/clip_distill loss':  clip_distill_loss.avg,
        } 


    comet_experiment.log_metrics(log)



    return totalloss.avg



def save_checkpoint(state, is_best, base_dir, filename="checkpoint.pth.tar"):
    torch.save(state, base_dir+filename)
    if is_best:
        shutil.copyfile(base_dir+filename, base_dir+"checkpoint_best_loss.pth.tar")

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-c",
        "--config",
        help="Path to config file",
    )

    given_configs, remaining = parser.parse_known_args(argv)
    with open(given_configs.config) as file:
        yaml_data= yaml.safe_load(file)
        parser.set_defaults(**yaml_data)

    parser.add_argument(
        '--exp_name', 
        type=str,
    )
    parser.add_argument(
        '--checkpoint', 
        type=str,
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
        args.num_workers = 32 if (os.cpu_count() - 2) > 32 else os.cpu_count() - 2
    experiment.log_parameters({"number of workers": args.num_workers})

    if args.PYTORCH_CUDA_ALLOC_CONF:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f"max_split_size_mb:{str(args.PYTORCH_CUDA_ALLOC_CONF)}"
    

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

    device = f"cuda:0" if args.cuda and torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")


    if args.dataset=='imagenet':
            
        train_transforms = transforms.Compose([
            transforms.Resize(256), 
            transforms.CenterCrop(256), 
            transforms.ToTensor()
        ])

        c = 12
        train_dataset = torchvision.datasets.ImageNet(args.dataset_path,split='train', transform=train_transforms)
        small_train_datasets = torch.utils.data.random_split(train_dataset,[106763]*c+[11])

        val_dataset = torchvision.datasets.ImageNet(args.dataset_path, split='val', transform=train_transforms)
        val_dataloader = DataLoader(val_dataset,batch_size=args.test_batch_size,num_workers=args.num_workers,shuffle=False,pin_memory=("cuda" in device), drop_last=False)
    else:
        raise NameError("Dataset not found")

    print(f'Dataset loaded, training dataset length: {len(train_dataset)}')

    rdcriterion       = RateDistortionLoss()
    clip_cls_criterion = ClipClsloss(args = args, device=device)
    out_features = 512

    in_features = 320
    ql = int(args.quality_level)

    adapter = Linear_Encoder(in_features=in_features, out_features=out_features, input_resolution=256, args=args)
    adapter = adapter.to(device)

    net = image_models[args.model](quality=ql)
    net = net.to(device)

    for k, p in net.named_parameters():
        p.requires_grad = False


    optimizer, aux_optimizer = configure_optimizers(adapter, args)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.optimizer['adapter']['Milestones'], gamma=0.1)


    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        logging.info("Loading "+str(args.checkpoint))
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
            if "base_coder." in k:
                name = k.replace("base_coder.", "")

            new_state_dict[name] = v


        new_state = {k: p for k, p in new_state_dict.items()}
        for k, p in net.named_parameters():
            # print(k)
            if (k not in new_state):
                if "SemanticLoss" in k or 'clip_model' in k:
                    continue
                print(f"No weight: {k}")
                continue
            if new_state[k].shape != p.shape:
                print(f"Size mismatch: {k}")
                if  not (args.TEST or args.state_dict_strict):
                    del new_state_dict[k]

            
        net.load_state_dict(new_state_dict, strict = True if args.state_dict_strict or args.resume_exp_key else False)

        if args.resume_exp_key is not None:
            last_epoch = checkpoint['epoch']+1
    
        if "adapter" in ckpt.keys():
            logging.info("Loading adapter")
            adapter.load_state_dict(ckpt['adapter'], strict = True)

        else:
            for k, p in adapter.named_parameters():
                print(f"No weight: {k}")

    else:
        print("Didn't load checkpoint")
    

    best_loss = float("inf")
    tqrange = tqdm.trange(last_epoch, args.epochs)

    if args.sanity and "LogLoss" not in args.exp_name:
        experiment.set_epoch(-1)
        loss = test_clipcls_epoch(-1, val_dataloader, net, rdcriterion, clip_cls_criterion, experiment, 'val', args = args, sanity=True, adapter = adapter)

        ckpt = {
                "epoch": -1,
                "state_dict": net.state_dict(),
                "adapter":adapter.state_dict(),
                "optimizer": optimizer.state_dict(), 
                "lr_scheduler": lr_scheduler.state_dict(), 
        }
        save_checkpoint(
                ckpt,
                False,
                base_dir, 
                filename='sanity_checkpoint.pth.tar'
        )


    for epoch in tqrange:
        train_dataloader = DataLoader(
                small_train_datasets[epoch%c],
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=True,
                pin_memory=("cuda" in device),
                drop_last=True
            )
        experiment.set_epoch(epoch)


        train_clipcls_one_epoch(
            model              = net,
            criterion_rd       = rdcriterion,
            criterion_clip_cls = clip_cls_criterion,
            train_dataloader   = train_dataloader,
            optimizer          = optimizer,
            aux_optimizer      = aux_optimizer,
            epoch              = epoch,
            comet_experiment   = experiment,
            args               = args,
            adapter            = adapter,
        )
        loss = test_clipcls_epoch(epoch, val_dataloader, net, rdcriterion, clip_cls_criterion, experiment, 'val', args = args, adapter = adapter)


        lr_scheduler.step()
        

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            ckpt = {
                "epoch": epoch,
                "state_dict": net.state_dict(),
                "adapter":adapter.state_dict(),
                "optimizer": optimizer.state_dict(), 
                "lr_scheduler": lr_scheduler.state_dict(), 
            }

            save_checkpoint(
                ckpt,
                is_best,
                base_dir,                
                filename='checkpoint.pth.tar'
            )
            if epoch % 5 == 0:
                shutil.copyfile(base_dir+'checkpoint.pth.tar', base_dir+ f"checkpoint_{str(epoch).zfill(3)}.pth.tar" )

if __name__ == "__main__":
    main(sys.argv[1:])