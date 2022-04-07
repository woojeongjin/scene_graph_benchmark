
import json
import os
import base64
import numpy as np
from tqdm import tqdm
from heuristic import heuristic
# import pickle

split = 'train2014'

ann_file = 'annotations/captions_'+split+'.json'
database = json.load(open(ann_file, 'r'))

# data_path = 'annotations/instances_val2017.json'
# ins = json.load(open(data_path, 'r'))

# per = json.load(open('annotations/pers'))
objects_vocab = dict()
img_ins = dict()
img_ins_lst = dict()

# with open('annotations_2017/instances_val2017.json', 'r') as f:
#     vocab = json.load(f)


# for voc in vocab['categories']:
#     objects_vocab[voc['id']] = voc['name'] 
# for voc in vocab['annotations']:
#     if voc['image_id'] not in img_ins:
#         img_ins[voc['image_id']]  = set()
#     img_ins[voc['image_id']].add(voc['category_id'])

# # print(img_ins.keys())  



# heuristic = heuristic(objects_vocab)

# captions = database['annotations'][:10]
# for caption in captions:
#     caption_text = caption['caption']
#     tags = img_ins[caption['image_id']]
#     match_dict = heuristic.compute(caption_text, tags)
#     tags_name = []
#     for tag in tags:
#         tags_name.append(objects_vocab[tag])
#     print(caption_text, match_dict, tags_name)


with open(split+'_objs.json', 'r') as f:
    objs = json.load(f)

for d in objs:
    image_id = int(d['image'].split('_')[-1].split('.')[0])
    if image_id not in img_ins:
        img_ins[image_id] = set()
        img_ins_lst[image_id] = []
    for cat in d['category']:
        img_ins[image_id].add(cat)
        img_ins_lst[image_id].append(cat)

heuristic = heuristic()


captions = database['annotations']
num_words = []
num_tags = []
num_align = []
output = dict()
for caption in tqdm(captions):

    caption_text = caption['caption']
    num_words.append(len(caption_text.split()))
    tags = list(img_ins[caption['image_id']])
    num_tags.append(len(img_ins_lst[caption['image_id']]))
    match_dict = heuristic.compute(caption_text, tags, True)
    for key in match_dict:
        match_dict[key] = list(match_dict[key])
    num_align.append(len(match_dict))


    result = {'id': caption['id'], 'img_id': caption['image_id'], 'caption': caption['caption'], 'tags': tags, 'match_dict': match_dict}
    
    if caption['image_id'] in output:
        output[caption['image_id']].append(result)
    else:
        output[caption['image_id']] = [result]
# print(output)

with open(split+'_align.json', 'w') as f:
    json.dump(output, f)

# print(np.mean(num_words), np.mean(num_tags), np.mean(num_align))

# def _load_json(path):
#     with open(path, 'r') as f:
#         return json.load(f)

# def b64_decode(string):
#     return base64.decodebytes(string.encode())

# def make_mapping(database):
#     final_dict = {}
#     total = 0
#     normal = 0
#     for idb in tqdm(enumerate(database)):
#         total+=1
#         try:
#             if os.path.exists(os.path.join(data_path, idb['id'] + ".json")):
#                 normal += 1
#                 frcnn_data = _load_json(os.path.join(data_path, idb['id'] + ".json"))
#                 boxes = np.frombuffer(b64_decode(frcnn_data['boxes']), dtype=np.int64).reshape((frcnn_data['num_boxes'], -1))
#                 boxes_cls_scores = np.frombuffer(b64_decode(frcnn_data['classes']), dtype=np.float32).reshape((frcnn_data['num_boxes'], -1))
#                 boxes_max_conf = boxes_cls_scores.max(axis=1)
#                 inds = np.argsort(boxes_max_conf)[::-1]
#                 boxes = boxes[inds]
#                 boxes_cls_scores = boxes_cls_scores[inds]
#                 captions = idb['captions']
#                 final_dict[idb['id']] = []
#                 for caption in captions:
#                     caption_text = caption['sentence']
#                     match_dict = heuristic.compute(caption_text, boxes, boxes_cls_scores)
#                     final_dict[idb['id']].append(match_dict)
#         except:
#             continue
#     print(normal, total)
#     return final_dict

# final_dict = make_mapping(database)



# with open('mapping.pkl', 'wb') as f:
#     pickle.dump(final_dict, f)

