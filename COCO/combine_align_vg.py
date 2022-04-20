import json
from tqdm import tqdm

vg_path = '/home/woojeong/VL-T5_eval/datasets/lxmert/vgnococo.json'
output_name = 'vgnococo_bbox.json'


vg_align_path = 'vg_bbox.json'

# vg_path = 'train2014_vg_bbox.json'

with open(vg_path, 'r') as f:
    vg = json.load(f)



with open(vg_align_path, 'r') as f:
    vg_align = json.load(f)

vg_vg = vg_align['vg']


new_data = []
for datum in tqdm(vg):
    # vg_bbox = vg_align[datum['img_id]]
    new_datum = dict()
    new_datum['img_id'] = datum['img_id']
    new_datum['sentf'] = dict()
    new_datum['sentf']['vg'] = []
    vg_sents = datum['sentf']['vg']

    # print('here')
    img_id = str(int(datum['img_id']))

    vg_sents = datum['sentf']['vg']
    vg_bboxes = vg_vg[img_id]
    for sent in vg_sents:
        for vg_bbox in vg_bboxes:
            if vg_bbox['caption'].lower().strip() == sent.lower().strip():
                bbox = vg_bbox['bbox']
                vg_size = vg_bbox['size']
                break
        new_datum['sentf']['vg'].append({'caption': sent, 'bbox': bbox, 'size': vg_size})


    new_data.append(new_datum)
        
with open(output_name, 'w') as f:
    json.dump(new_data, f, indent=4)