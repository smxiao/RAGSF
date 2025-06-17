import json
import os
from config import get_params
from openprompt.data_utils.data_processor import DataProcessor
from openprompt.data_utils.utils import InputExample
import re

args = get_params()

domains = ["AddToPlaylist", "BookRestaurant", "GetWeather", "PlayMusic", "RateBook", "SearchCreativeWork", "SearchScreeningEvent"]
domain2slot = {
    "AddToPlaylist": ['music_item', 'playlist_owner', 'entity_name', 'playlist', 'artist'],
    "BookRestaurant": ['city', 'facility', 'timeRange', 'restaurant_name', 'country', 'cuisine', 'restaurant_type', 'served_dish', 'party_size_number', 'poi', 'sort', 'spatial_relation', 'state', 'party_size_description'],
    "GetWeather": ['city', 'state', 'timeRange', 'current_location', 'country', 'spatial_relation', 'geographic_poi', 'condition_temperature', 'condition_description'],
    "PlayMusic": ['genre', 'music_item', 'service', 'year', 'playlist', 'album','sort', 'track', 'artist'],
    "RateBook": ['object_part_of_series_type', 'object_select', 'rating_value', 'object_name', 'object_type', 'rating_unit', 'best_rating'],
    "SearchCreativeWork": ['object_name', 'object_type'],
    "SearchScreeningEvent": ['timeRange', 'movie_type', 'object_location_type','object_type', 'location_name', 'spatial_relation', 'movie_name']
}

def snips_data_reader(n_samples=0):
    dataset = {}
    snips_process = SnipsProcessor()

    dataset["train"] = []
    dataset["val"] = []
    dataset["test"] = []
    
    print(args.target_domain)
    assert args.target_domain in domains
    n_samples = args.n_samples
    
    source_domain_dataset = snips_process.get_examples(data_dir=args.dataset_dir+args.target_domain+'/', target_domain=args.target_domain, istarget_domain=False)
    dataset["train"].extend(source_domain_dataset)

    val_test_dataset = snips_process.get_examples(data_dir=args.dataset_dir+args.target_domain+'/', target_domain=args.target_domain, istarget_domain=True)
    num_val = 500 * len(domain2slot[args.target_domain])
    num_train = n_samples * len(domain2slot[args.target_domain]) # default:zero-shot n_samples=0, num_train=0
    dataset["train"].extend(val_test_dataset[:num_train])
    dataset["val"].extend(val_test_dataset[num_train:num_val])  # dev set
    dataset["test"].extend(val_test_dataset[num_val:]) # test set
    
    return dataset



class SnipsProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.labels = None

    def get_examples(self, data_dir: str, target_domain: str, istarget_domain: bool):
        instances = []
        if istarget_domain: # read target_domain data
            file_path = os.path.join(data_dir, f"target_{target_domain}.json")
            target_labels = domain2slot[target_domain]
            target_labels_str = ' '.join(target_labels)
            with open(file_path) as f:
                data = json.load(f)
                for i, di in enumerate(data):
                    guid = str(i)
                    sentence = di["sentence"]
                    examples_li = di["examples"]
                    eg = []
                    for e in examples_li:
                        eg.append(e["example"])
                    
                    text_a = f"Slots in {target_domain} domain: {target_labels_str}. Examples: example #1: {eg[0]}. Query: {sentence}"
                    text_b = di["slot_type"]
                    entity = di["entity"]
                    instance = InputExample(guid=guid,
                                            text_a=text_a,
                                            text_b=text_b,
                                            tgt_text=entity)
                    instances.append(instance)
        else:
            for sf in os.listdir(data_dir):
                if ("target" in sf): #remove target_domain
                    continue
                else:
                    cur_domain = re.findall("source_(.*?).json", sf)[0]
                    cur_domain_labels = domain2slot[cur_domain]
                    cur_domain_labels_str = ' '.join(cur_domain_labels)
                    with open(os.path.join(data_dir, sf)) as f:
                        data = json.load(f)
                        for i, di in enumerate(data):
                            guid = str(i)
                            sentence = di["sentence"]
                            examples_li = di["examples"]
                            eg = []
                            for e in examples_li:
                                eg.append(e["example"])
                            text_a = f"Slots in {cur_domain} domain: {cur_domain_labels_str}. Examples: example #1: {eg[0]}. Query: {sentence}"                           
                            text_b = di["slot_type"]
                            entity = di["entity"]
                            instance = InputExample(guid=guid,
                                                    text_a=text_a,
                                                    text_b=text_b,
                                                    tgt_text=entity)
                            instances.append(instance)

        return instances
        
        
