from tqdm import tqdm
from collections import defaultdict
from config import get_params

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


def calcualate_f1_score(tp, fp, fn):
    p = 0 if tp + fp == 0 else 1.*tp / (tp + fp)
    r = 0 if tp + fn == 0 else 1.*tp / (tp + fn)
    f = 0 if p + r == 0 else 2 * p * r / (p + r)
    print(f"precision is {p} , recall is {r}")
    return f


def universal_evaluate_snips(prompt_model, dataloader):
    '''
    calcualate average f1 score
    '''
    generated_sentence = []  # predict span
    groundtruth_sentence = [] # gold span
    
    gold_seen = 0
    gold_unseen = 0
    predict_seen = 0
    predict_unseen = 0
    correct_seen = 0
    correct_unseen = 0
    
    gold_dict = defaultdict(int)
    predict_dict = defaultdict(int)
    correct_dict =defaultdict(int)
    
    slot_list = domain2slot[args.target_domain]
    prompt_model.eval()
    
    pbar = tqdm(enumerate(dataloader),total=len(dataloader))
    for step, inputs in pbar:
        inputs = inputs.cuda()
        _, output_sentence = prompt_model.generate(inputs)
        assert len(output_sentence)==len(inputs['tgt_text'])

        for i,count in enumerate(zip(inputs['tgt_text'], output_sentence)):
            cur_slot = slot_list[i]
            if count[0]!="none":
                gold_dict[cur_slot] +=1
            if count[1]!="none":
                predict_dict[cur_slot] +=1
            if count[0]==count[1]!="none":
                correct_dict[cur_slot] +=1


    guess_trunk_nums = sum(predict_dict.values())
    true_trunk_nums = sum(gold_dict.values())
    correct_trunk_nums = sum(correct_dict.values())
    
    tp = correct_trunk_nums
    fp = guess_trunk_nums-tp
    fn = true_trunk_nums-tp
    
    print(f"total number of slots is {true_trunk_nums}")
    print(f"guessed number of slots is {guess_trunk_nums}")
    print(f"correct number of guessed slots is {correct_trunk_nums}")
    
    return calcualate_f1_score(tp,fp,fn)


