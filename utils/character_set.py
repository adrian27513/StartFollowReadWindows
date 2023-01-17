import sys
import json
import os
from collections import defaultdict

def load_char_set(char_set_path):
    with open(char_set_path, encoding='utf-8') as f:
        char_set = json.load(f)

    idx_to_char = {}
    for k,v in char_set['idx_to_char'].items():
        idx_to_char[int(k)] = v

    return idx_to_char, char_set['char_to_idx']

if __name__ == "__main__":
    character_set_path = "../data/char_set.json"
    out_char_to_idx = {}
    out_idx_to_char = {}
    char_freq = defaultdict(int)
    data_list = ["../data/train_a_training_set.json", "../data/train_a_validation_set.json", "../data/train_b_training_set.json", "../data/train_b_validation_set.json"]
    # data_list = ["../data/train_a_validation_set.json"]
    for i in range(len(data_list)):
        data_file = data_list[i]
        with open(data_file, encoding='utf-8') as f:
            paths = json.load(f)

        print(data_file)
        for json_path, image_path in paths:
            json_path = "../" + json_path
            with open(json_path, encoding='utf-8') as f:
                data = json.load(f)
            cnt = 1 # this is important that this starts at 1 not 0
            for i, data_item in enumerate(data):
                for c in data_item.get('gt', None):
                    if c is None:
                        print("There was a None GT")
                        continue
                    if c not in out_char_to_idx:
                        out_char_to_idx[c] = cnt
                        out_idx_to_char[cnt] = c
                        cnt += 1
                        if json_path.split("\\")[-1] == "000046.json" and i == 1:
                            print(c)
                            # print(data)
                            print(data_item)
                            print(json_path)
                            print("---")
                    char_freq[c] += 1


    out_char_to_idx2 = {}
    out_idx_to_char2 = {}

    for i, c in enumerate(sorted(out_char_to_idx.keys())):
        out_char_to_idx2[c] = i+1
        out_idx_to_char2[i+1] = c

    output_data = {
        "char_to_idx": out_char_to_idx2,
        "idx_to_char": out_idx_to_char2
    }

    for k,v in sorted(iter(char_freq.items()), key=lambda x: x[1]):
        print(k, v)

    print(("Size:", len(output_data['char_to_idx'])))

    with open(character_set_path, 'w') as outfile:
        json.dump(output_data, outfile)
