import json
from tqdm import tqdm

mscoco_path = '/home/woojeong/VL-T5_eval/datasets/lxmert/mscoco_resplit_train.json'
output_name = 'mscoco_resplit_train_bbox.json'

# mscoco_path = '/home/woojeong/VL-T5_eval/datasets/lxmert/mscoco_resplit_val.json'
# output_name = 'mscoco_resplit_val_bbox.json'


align_train_path = 'train2014_token_bbox.json'
align_val_path = 'val2014_token_bbox.json'
vg_path = 'vg_bbox.json'

# vg_path = 'train2014_vg_bbox.json'

with open(mscoco_path, 'r') as f:
    mscoco = json.load(f)

with open(align_train_path, 'r') as f:
    align = json.load(f)

with open(align_val_path, 'r') as f:
    align_val = json.load(f)

with open(vg_path, 'r') as f:
    vg = json.load(f)

vg_coco = vg['mscoco']
vg_vg = vg['vg']


new_data = []
for datum in tqdm(mscoco):
    if datum['img_id'] in align.keys():
        token_bboxes = align[datum['img_id']]
    else:
        token_bboxes = align_val[datum['img_id']]
    # vg_bbox = vg_align[datum['img_id]]
    new_datum = dict()
    new_datum['img_id'] = datum['img_id']
    new_datum['sentf'] = dict()
    new_datum['sentf']['mscoco'] = []
    coco_sents = datum['sentf']['mscoco']

    for sent in coco_sents:
        for token_bbox in token_bboxes:
            if token_bbox['caption'].lower().strip() == sent.lower().strip():
                bbox = token_bbox['token_bbox']
                break
        new_bbox = dict()
        for key, value in bbox.items():
            x, y, xm, ym = value
            w = float(xm)-float(x)
            h = float(ym)-float(y)
            if w <= 0 :
                w = 0
            if h <= 0:
                h = 0
            if x <=0 :
                x = 0
            if y <= 0 :
                y = 0
            new_bbox[key] = [x, y, w, h]

        new_datum['sentf']['mscoco'].append({'caption': sent, 'token_bbox': new_bbox})

    

    if 'vg' in datum['sentf'].keys():
        # print('here')
        new_datum['sentf']['vg'] = []
        img_id = str(int(datum['img_id'].split('_')[-1]))

        vg_sents = datum['sentf']['vg']
        vg_bboxes = vg_coco[img_id]
        for sent in vg_sents:
            for vg_bbox in vg_bboxes:
                if vg_bbox['caption'].lower().strip() == sent.lower().strip():
                    bbox_vg = vg_bbox['bbox']
                    vg_size = vg_bbox['size']
                    break
            new_datum['sentf']['vg'].append({'caption': sent, 'bbox': bbox_vg, 'size': vg_size})


    new_data.append(new_datum)
        
with open(output_name, 'w') as f:
    json.dump(new_data, f, indent=4)