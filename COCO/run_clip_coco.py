import torch
import clip
from PIL import Image
import json
from tqdm import tqdm

split = 'val2014'
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)

with open(split+'_objs.json', 'r') as f:
    objs = json.load(f)

with open(split+'_align.json', 'r') as f:
    aligns = json.load(f)


objs_dict = dict()
for obj in objs:
    img = obj['image']
    img_id = int((img.split('.')[0]).split('_')[-1])
    objs_dict[img_id] = obj


def reverse_dict(match_dict):
    token_dict = dict()
    for tag, toks in match_dict.items():
        for tok in toks:
            if tok in token_dict.keys():
                if tag not in token_dict[tok]:
                    token_dict[tok].append(tag)
            else:
                token_dict[tok] = [tag]

    return token_dict
        


output = dict()
for img_id in tqdm(aligns):
    data = aligns[img_id]
    img_id = data[0]['img_id']
    objs = objs_dict[img_id]
    category = objs['category']
    bbox = objs['bbox']
    img = Image.open("Image/"+split+"/"+objs['image'])
    filename = objs['image'].split('.')[0]
    output[filename] = []
    for point in data:
        caption = point['caption']
        match_dict = point['match_dict'] # tag: token
        token_dict = reverse_dict(match_dict)
        # print(match_dict)
        # print(token_dict)

        token_bbox = dict()

        sent_tokens = []
        for i, tok in enumerate(caption.split()):
            tok = tok.lower().strip('.,?;:')
            sent_tokens.append(tok)

        for tok, tags in token_dict.items():
            tag_bbox = []
            cropped_imgs = []
            for tag in tags:
                for i, cat in enumerate(category):
                    if cat == tag:
                        cropped_imgs.append(img.crop(bbox[i]))
                        tag_bbox.append(bbox[i])
            
            
            if len(cropped_imgs) == 1:
                token_bbox[tok] = tag_bbox[0]
            else:
                sz = 2
                index = sent_tokens.index(tok)
                surrounding = [" ".join(sent_tokens[index-sz: index+sz+1])]

                images = []
                for cropped_img in cropped_imgs:
                    images.append(preprocess(cropped_img).unsqueeze(0).to(device))
                text = clip.tokenize(surrounding).to(device)
                print(len(images), len(cropped_imgs), len(tag_bbox), tags, category)
                image = torch.cat(images)

                with torch.no_grad():
                    
                    logits_per_image, logits_per_text = model(image, text)
                    img_ind = torch.argmax(logits_per_text.reshape(-1))
                    # print(logits_per_text, img_ind)
                    token_bbox[tok] = tag_bbox[img_ind.item()]

                    # print(token_bbox)
        output[filename].append({'caption': caption, 'token_bbox': token_bbox})


        # for tag, tokens in match_dict.items():
        #     # find bbox
        #     tag_bbox = []
        #     cropped_imgs = []
        #     for i, cat in enumerate(category):
        #         if cat == tag:
        #             cropped_imgs.append(img.crop(bbox[i]))
        #             tag_bbox.append(bbox[i])

        #     multiple_bbox = False
        #     multiple_tokens = False
        #     repeated_tokens = False
        #     if len(cropped_imgs) > 1:
        #         multiple_bbox = True
        #     if len(tokens) > 1:
        #         multiple_tokens = True

        #     for tok in tokens:
        #         assert tok in sent_tokens
            
        #     if multiple_bbox is False and multiple_tokens is False:
        #         token_bbox[tokens[0]] = cropped_imgs[0]
        #     else:
        #         surroundings = []
        #         indices = []
        #         for tok in tokens:
        #             indices.append(sent_tokens.index(tok))
                
        #         sz = 2
        #         for ind in indices:
        #             surrounding = " ".join(sent_tokens[ind-sz: ind+sz+1])
        #             surroundings.append(surrounding)
                
        #         images = []
        #         for cropped_img in cropped_imgs:
        #             images.append(preprocess(cropped_img).unsqueeze(0).to(device))

        #         text = clip.tokenize(surroundings).to(device)
        #         image = torch.cat(images)

        #         with torch.no_grad():
        #             # image_features = model.encode_image(image)
        #             # text_features = model.encode_text(text)
                    
        #             logits_per_image, logits_per_text = model(image, text)

        #         for j, logits in enumerate(torch.transpose(logits_per_text, 0, 1)):
        #             img_ind = torch.argmax(logits)
        #             token_bbox[token[j]] = tag_bbox[img_ind]


        



with open(split+'_token_bbox.json', 'w') as f:
    json.dump(output, f, indent=4)





# cropped_img =Image.open("CLIP.png").crop(bbox)
# image = preprocess(cropped_img).unsqueeze(0).to(device)
# text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

# with torch.no_grad():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)
    
#     logits_per_image, logits_per_text = model(image, text)
#     probs = logits_per_image.softmax(dim=-1).cpu().numpy()