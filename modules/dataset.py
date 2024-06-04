import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import BertTokenizer, ViTImageProcessor
from io import BytesIO

class MNERProcessor:
    def __init__(self, args) -> None:
        self.data_path = args.data_path
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
    
    def load_from_file(self, file_name):
        words, labels, images = [], [], []
        with open(os.path.join(self.data_path, file_name), 'r', encoding='utf8') as f:
            for i, line in enumerate(f):
                line_json = eval(line)
                word = line_json["text"]
                label = line_json["label"]
                for j, l in enumerate(label):
                    # entity label is MISC in twitter2017, but MIS in MNER-MI
                    if l == "B-MISC":
                        label[j] = "B-MIS"
                    elif l == "I-MISC":
                        label[j] = "I-MIS"
                image = line_json["images"] 
                words.append(word)
                labels.append(label)
                images.append(image)
        
        assert len(words) == len(labels) == len(images)      
        
        return {"words": words, "labels": labels, "images": images} 

    # use softmax as decoder
    def get_label_mapping(self):
        LABEL_LIST = ["O", "B-MIS", "I-MIS", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
        label_mapping = {label: idx for idx, label in enumerate(LABEL_LIST, 0)}
        return label_mapping
    # use crf as decoder 
    def get_label_crf_mapping(self):
        LABEL_LIST = ["O", "B-MIS", "I-MIS", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X"]
        label_mapping = {label: idx for idx, label in enumerate(LABEL_LIST, 0)}
        return label_mapping

# Process each image separately, and use crf as decoder, and VIT as image encoder
class TextSperateImageVITCRFDataset(Dataset):
    def __init__(self, args, file_name) -> None:
        self.max_seq_length = args.max_seq_length
        self.processor = MNERProcessor(args)
        # VIT processor for VIT encoder
        self.vit_processor = ViTImageProcessor.from_pretrained(args.vit_model)
        self.data_dict = self.processor.load_from_file(file_name)
        self.tokenizer = self.processor.tokenizer
        self.label_mapping = self.processor.get_label_crf_mapping()
        # MNER-MI image path
        self.image_path = args.image_path
        # twitter-2017 image path (if use MNER-MI-Plus dataset)
        self.twitter2017_image_path = args.twitter2017_image_path
        
    def __len__(self):
        return len(self.data_dict["words"])

    def __getitem__(self, idx):
        word_list = self.data_dict["words"][idx]
        label_list = self.data_dict["labels"][idx]
        images = self.data_dict["images"][idx]
        
        # (4, 3, 224, 224); initialize the image representation with all zero tensors
        seperate_images_feature = torch.zeros((4, 3, 224, 224))
        for i, image in enumerate(images):
            if "twitter2017" in image:
                img_path = os.path.join(self.twitter2017_image_path, image.split('-')[1])
            else:
                img_path = os.path.join(self.image_path, image)
            try:
                image = Image.open(img_path).convert("RGB")
                pixels = self.vit_processor(images=image, return_tensors='pt')["pixel_values"]
                seperate_images_feature[i] = pixels
            except OSError:
                try:
                    with open(img_path, 'rb') as f:
                        # some images is corrupted and missing JPEG end markers, so manually add \xff and \xd9
                        f = f.read()
                        f=f+B'\xff'+B'\xd9'
                    image = Image.open(BytesIO(f)).convert("RGB")
                    pixels = self.vit_processor(images=image, return_tensors='pt')["pixel_values"]
                    seperate_images_feature[i] = pixels
                except:
                    # if the image does not exist, the altername image is used
                    img_path = os.path.join(self.image_path, 'inf.png')
                    image = Image.open(img_path).convert("RGB")
                    pixels = self.vit_processor(images=image, return_tensors='pt')["pixel_values"]
                    seperate_images_feature[i] = pixels


        tokens, labels = [], []
        for i, word in enumerate(word_list):
            # a word may contain multiple tokens
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            label = label_list[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(self.label_mapping[label])
                else:
                    # the extra tokends labels are X
                    labels.append(self.label_mapping["X"])
        if len(tokens) >= self.max_seq_length - 1:
            tokens = tokens[0:(self.max_seq_length - 2)]
            labels = labels[0:(self.max_seq_length - 2)]

        encode_dict = self.tokenizer.encode_plus(tokens, max_length=self.max_seq_length, truncation=True,
                                                 padding='max_length')
        input_ids, token_type_ids, attention_mask = encode_dict['input_ids'], encode_dict['token_type_ids'], \
                                                    encode_dict['attention_mask']
        labels = [self.label_mapping["X"]] + labels + [self.label_mapping["X"]] + [self.label_mapping["O"]] * (self.max_seq_length - len(labels) - 2) 
        
        return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask), \
                seperate_images_feature, torch.tensor(labels) 


