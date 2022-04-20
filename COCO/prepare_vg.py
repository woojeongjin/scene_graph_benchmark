import json
from unittest import result

# x,y,w,h

with open('VG/region_descriptions.json' ,'r') as f:
    captions = json.load(f)

with open('VG/image_data.json', 'r') as f:
    metas = json.load(f)


output_vg = dict()
output_coco = dict()

for datum, meta in zip(captions, metas):

    result = []
    for region in datum['regions']:
        caption = region['phrase']
        bbox = [region['x'], region['y'], region['width'], region['height']]

        if int(region['x']) ==713:
            print(meta['width'], bbox) 

        result.append({'caption': caption, 'bbox': bbox, 'size':(meta['width'], meta['height'])})


    if meta['coco_id'] is not None:
        output_coco[int(meta['coco_id'])] = result
    else:
        output_vg[int(meta['image_id'])] = result


with open('vg_bbox.json', 'w') as f:
    json.dump({'vg': output_vg, 'mscoco': output_coco}, f, indent=4)