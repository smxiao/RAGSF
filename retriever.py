from itertools import zip_longest
import json
import os
import pickle
import re
import time
import numpy as np
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

domain2slot = {
    "AddToPlaylist": ['music_item', 'playlist_owner', 'entity_name', 'playlist', 'artist'],
    "BookRestaurant": ['city', 'facility', 'timeRange', 'restaurant_name', 'country', 'cuisine', 'restaurant_type', 'served_dish', 'party_size_number', 'poi', 'sort', 'spatial_relation', 'state', 'party_size_description'],
    "GetWeather": ['city', 'state', 'timeRange', 'current_location', 'country', 'spatial_relation', 'geographic_poi', 'condition_temperature', 'condition_description'],
    "PlayMusic": ['genre', 'music_item', 'service', 'year', 'playlist', 'album','sort', 'track', 'artist'],
    "RateBook": ['object_part_of_series_type', 'object_select', 'rating_value', 'object_name', 'object_type', 'rating_unit', 'best_rating'],
    "SearchCreativeWork": ['object_name', 'object_type'],
    "SearchScreeningEvent": ['timeRange', 'movie_type', 'object_location_type','object_type', 'location_name', 'spatial_relation', 'movie_name']
}

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
dataset_path = './raw_data_snips/'
emb_cache_path = './emb_cache/'
retrieve_dataset_path = './retrieve_data/'
target_domain = None
topk = 1

# embedding pkl
if not os.path.exists(emb_cache_path):
    os.makedirs(emb_cache_path)
    for file in os.listdir(dataset_path):
        file_path = os.path.join(dataset_path, file)
        domain_name = re.findall("universal_(.*?).json", file)[0]
        print(file_path)
        sentences = []
        sents = []
        with open(file_path) as fIn:
            data = json.load(fIn)
            n = len(data)
            for di in data:
                utt = di["sentence"]
                slot_list = di["slot_types"]
                entity_list = di["entities"]
                for x, y in zip(slot_list, entity_list):
                    s_temp = domain_name+"###"+utt+". "+x+" : "+y
                    sentences.append(s_temp)
                    s_temp2 = utt + ". "+x # w/o y=value
                    sents.append(s_temp2)
        assert len(sentences) == len(sents)
        print(f"\n Encode {file} <{n}> sentences. This might take a while...")
        sentences_emb = model.encode(sents, batch_size=8, show_progress_bar=True, convert_to_numpy=True)
        print("Store file on disc...")
        file_emb_path = f"{emb_cache_path}{file}.pkl"
        with open(file_emb_path, "wb+") as fOut:
            pickle.dump({"sentences": sentences, "embeddings": sentences_emb}, fOut)
        print(f"{file} Finished*******")

# construct data with examples
start_time = time.time()

domain_list = domain2slot.keys()
for d in domain_list:
    if not os.path.exists(f"{retrieve_dataset_path}{d}"):
        os.makedirs(f"{retrieve_dataset_path}{d}")
for domain in domain_list:
    target_domain = domain
    print(f"*******target domain: {target_domain}*******")
    for res in domain_list:
        if res == target_domain:
            retrieve_sents = []
            retrieve_embs = []
            print("Load pre-computed embeddings from disc...")
            count = 0
            for ef in os.listdir(emb_cache_path):
                    if (target_domain in ef): # remove target_domain
                        continue
                    else:
                        count += 1
                        with open(os.path.join(emb_cache_path, ef), "rb") as emd_f:
                            cache_data = pickle.load(emd_f)
                            retrieve_sents.extend(cache_data["sentences"])
                            retrieve_embs.extend(cache_data["embeddings"])
            retrieve_embs = retrieve_embs / np.linalg.norm(retrieve_embs, axis=1)[:, None]
            number = len(retrieve_embs)
            print("Corpus loaded with {} sentences / embeddings".format(len(retrieve_sents)))
            assert count == (len(domain_list)-1)
            cur_data_path = f"{dataset_path}universal_{res}.json"
            cur_data_example = []
            print(f"constructing...")
            with open(cur_data_path) as cur_f:
                cur_data = json.load(cur_f) 
                for i in tqdm(range(len(cur_data)), desc = 'Retrieve and Construct'):
                    di = cur_data[i]
                    utt = di["sentence"]
                    slot_list = di["slot_types"]
                    entity_list = di["entities"]
                    for x, y in zip(slot_list, entity_list):
                        examples = []
                        input_ = utt + ". "+x
                        input_emb = model.encode(input_)
                        input_emb = input_emb / np.linalg.norm(input_emb)
                        input_emb = np.expand_dims(input_emb, axis=0)
                        correct_res = util.semantic_search(input_emb, retrieve_embs, top_k=topk)[0]
                        assert len(retrieve_embs) == number
                        correct_res_ids = []
                        correct_res_scores = []
                        for r in correct_res:
                            correct_res_ids.append(r["corpus_id"])
                            correct_res_scores.append(r["score"])
                        assert len(correct_res_ids) == len(correct_res_scores)
                        for id, score in zip(correct_res_ids, correct_res_scores):
                            tmp = retrieve_sents[int(id)].split("###")
                            eg = {"score": score, "domain": tmp[0], "example": tmp[1]}
                            examples.append(eg)
                        instance = {"sentence": utt, "examples": examples, "slot_type": x, "entity": y}
                        cur_data_example.append(instance)
            num = len(cur_data_example)
            r_path = f"{retrieve_dataset_path}{target_domain}/target_{res}.json"
            with open(r_path, "w") as rf:
                json.dump(cur_data_example, rf, indent=4)
            print(f"{res} Finished*****")
        else:
            print(f"current domain: {res}")
            # remove target_domain and current_domain
            retrieve_sents = []
            retrieve_embs = []
            print("Load pre-computed embeddings from disc...")
            count = 0
            for ef in os.listdir(emb_cache_path):
                    if (res in ef) or (target_domain in ef): # remove target_domain
                        continue
                    else:
                        count += 1
                        with open(os.path.join(emb_cache_path, ef), "rb") as emd_f:
                            cache_data = pickle.load(emd_f)
                            retrieve_sents.extend(cache_data["sentences"])
                            retrieve_embs.extend(cache_data["embeddings"])
            retrieve_embs = retrieve_embs / np.linalg.norm(retrieve_embs, axis=1)[:, None]
            print("Corpus loaded with {} sentences / embeddings".format(len(retrieve_sents)))
            number2 = len(retrieve_embs)
            assert count == (len(domain_list)-2)
            cur_data_path = f"{dataset_path}universal_{res}.json"
            cur_data_example = []
            print(f"constructing...")
            with open(cur_data_path) as cur_f:
                cur_data = json.load(cur_f) 
                for i in tqdm(range(len(cur_data)), desc = 'Retrieve and Construct'):
                    di = cur_data[i]
                    utt = di["sentence"]
                    slot_list = di["slot_types"]
                    entity_list = di["entities"]
                    for x, y in zip(slot_list, entity_list):
                        examples = []
                        input_ = utt + ". "+x
                        input_emb = model.encode(input_)
                        input_emb = input_emb / np.linalg.norm(input_emb)
                        input_emb = np.expand_dims(input_emb, axis=0)
                        correct_res = util.semantic_search(input_emb, retrieve_embs, top_k=topk)[0]
                        assert len(retrieve_embs) == number2
                        correct_res_ids = []
                        correct_res_scores = []
                        for r in correct_res:
                            correct_res_ids.append(r["corpus_id"])
                            correct_res_scores.append(r["score"])
                        assert len(correct_res_ids) == len(correct_res_scores)
                        for id, score in zip_longest(correct_res_ids, correct_res_scores):
                            tmp = retrieve_sents[int(id)].split("###")
                            eg = {"score": score, "domain": tmp[0], "example": tmp[1]}
                            examples.append(eg)
                        instance = {"sentence": utt, "examples": examples, "slot_type": x, "entity": y}
                        cur_data_example.append(instance)
            num = len(cur_data_example)
            r_path = f"{retrieve_dataset_path}{target_domain}/source_{res}.json"
            with open(r_path, "w") as rf:
                json.dump(cur_data_example, rf, indent=4)
            print(f"{res} Finished*****")

end_time = time.time()
print("Results (after {:.3f} seconds):".format(end_time - start_time))
            
                

                       
                        
            

