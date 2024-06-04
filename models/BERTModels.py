import torch
from torch import nn
from transformers import BertModel, ViTModel
from torchcrf import CRF

class TPM_MI(torch.nn.Module):
    def __init__(self, label_list, args):
        super().__init__()
        self.num_labels = len(label_list)
        # text encoder
        self.bert = BertModel.from_pretrained(args.bert_model)
        # image encoder
        self.vit = ViTModel.from_pretrained(args.vit_model)
        self.config = self.bert.config
        self.n_layer = self.config.num_hidden_layers
        self.n_head = self.config.num_attention_heads
        self.n_embd = self.config.hidden_size // self.config.num_attention_heads

        # project images as prompts
        self.img_prompt_encoder = torch.nn.Sequential(
            nn.Linear(in_features=768, out_features=1024),
            nn.Tanh(),
            nn.Linear(in_features=1024, out_features=self.n_layer * 2 * self.config.hidden_size)
        )
        # learnable postional embeddings
        self.temporalEmbedding = torch.nn.Embedding(4, 768)
        self.transformer = nn.TransformerEncoderLayer(d_model=768, nhead=8)
        self.fc = torch.nn.Linear(768, self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)

    
    def get_prompt(self, img_features=None):
        bsz, img_len, _ = img_features.size()
        # (bsz, 4, 768)-> (bsz, 4, n_layer*2*hidden_size)
        past_key_values_vit = self.img_prompt_encoder(img_features)
        past_key_values_vit = past_key_values_vit.view(
            bsz,
            img_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )

        # (bsz, seq_len + seq_len2, 12*2, n_head(12), 64)
        past_key_values = past_key_values_vit
        # (12*2, bsz, n_head, len, 64)
        # 12*[2,bsz,n_head,len,64]
        past_key_values = past_key_values.permute([2,0,3,1,4]).split(2)

        return past_key_values
    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                 img_features=None, labels=None):
        bsz, img_len, channels, height, width = img_features.shape
        # (bsz, 4, 3, 224, 224) -> (bsz*4, 3, 224, 224)
        pixel_values = img_features.reshape(bsz*img_len, channels, height, width)
        # (bsz*4, 3, 224, 224) -> (bsz*4, 768)
        img_features = self.vit(pixel_values=pixel_values)[1]
        # (bsz*4, 768) -> (bsz, 4, 768)
        img_features = img_features.reshape(bsz, img_len, -1)
        temp_embeddings = self.temporalEmbedding(torch.arange(4).to(img_features.device))
        # (4, 768) -> (1, 4, 768)
        temp_embeddings = temp_embeddings.unsqueeze(0)
        img_features = img_features + temp_embeddings
        # (bsz, 4, 768)
        tempral_features = self.transformer(img_features)
        # img_features (bsz, 4, 768) -> past_key_values 12*[2, bsz, n_head, len, 64] 
        past_key_values = self.get_prompt(tempral_features)
        prompt_guids_length = past_key_values[0][0].size(2)
        bsz = attention_mask.size(0)
        prompt_guids_mask = torch.ones((bsz, prompt_guids_length)).to(attention_mask.device)
        prompt_attention_mask = torch.cat((prompt_guids_mask, attention_mask), dim=1)
        
        output_bert_result = self.bert(input_ids=input_ids,
                                       attention_mask=prompt_attention_mask,
                                       token_type_ids=token_type_ids,
                                       past_key_values=past_key_values)
        sequence_output = output_bert_result[0]
        emissions = self.fc(sequence_output)
        logits = self.crf.decode(emissions, attention_mask.byte())

        if labels is not None:
            loss = - self.crf(emissions, labels, mask=attention_mask.byte(), reduction='mean')
            return (logits, loss)
