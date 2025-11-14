import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math
from transformers import AutoTokenizer


# --- Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# --- ResNet-50 Image Encoder with Spatial Features ---
class ResNetImageEncoder(nn.Module):
    def __init__(self, pretrained=True, freeze_epochs=5):
        
        super().__init__()
        resnet = models.resnet50(pretrained=pretrained)
        
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        self.channel_reduction = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0)
        
        self.freeze_epochs = freeze_epochs
        self.current_epoch = 0
        self._freeze_backbone()
        
        print(f"ResNet-50 Image Encoder initialized (pretrained={pretrained})")
        print(f"Output feature map: (batch, 512, 7, 7)")

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("ResNet backbone frozen")

    def _unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("ResNet backbone unfrozen")

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        if epoch < self.freeze_epochs:
            if any(p.requires_grad for p in self.backbone.parameters()):
                self._freeze_backbone()
        else:
            if not any(p.requires_grad for p in self.backbone.parameters()):
                self._unfreeze_backbone()

    def forward(self, img):
       
        features = self.backbone(img)
        
        features = self.channel_reduction(features)
        
        return features


# --- Image Spatial Attention ---
class SpatialAttention(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        self.conv = nn.Conv2d(d_model, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        attention_map = self.sigmoid(self.conv(x))
        attended = x * attention_map
        return attended, attention_map


# --- Text Encoder (for report embeddings) ---
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, max_length=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_length)
        self.d_model = d_model

    def forward(self, text_ids):
        
        x = self.embedding(text_ids) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        return x


# --- Transformer Decoder with Cross-Attention ---
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=512, nhead=4, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.activation = nn.ReLU()

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, 
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        
        # Self-attention
        tgt2 = self.norm1(tgt)
        tgt2, _ = self.self_attn(tgt2, tgt2, tgt2, attn_mask=tgt_mask, 
                                  key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        
        # Cross-attention to image features
        tgt2 = self.norm2(tgt)
        tgt2, attn_weights = self.cross_attn(tgt2, memory, memory, 
                                              attn_mask=memory_mask,
                                              key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        
        # Feed-forward
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        
        return tgt, attn_weights


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=4, num_layers=2, 
                 dim_feedforward=2048, dropout=0.1, max_length=512):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Word embedding and positional encoding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_length)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        print(f"Transformer Decoder: d_model={d_model}, nhead={nhead}, layers={num_layers}")

    def forward(self, tgt_ids, img_features, tgt_mask=None, tgt_key_padding_mask=None):
        
        # Embed and add positional encoding
        tgt = self.embedding(tgt_ids) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        
        # Pass through decoder layers
        attn_weights_list = []
        for layer in self.layers:
            tgt, attn_weights = layer(tgt, img_features, tgt_mask=tgt_mask,
                                      tgt_key_padding_mask=tgt_key_padding_mask)
            attn_weights_list.append(attn_weights)
        
        # Project to vocabulary
        logits = self.output_proj(tgt)
        
        return logits, attn_weights_list

    def generate_square_subsequent_mask(self, sz: int):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
        return mask


class TienetReportGenerator(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=4, num_decoder_layers=2,
                 dim_feedforward=2048, dropout=0.1, freeze_epochs=5):
        super().__init__()
        self.d_model = d_model
        
        self.image_encoder = ResNetImageEncoder(pretrained=True, freeze_epochs=freeze_epochs)
        
        self.spatial_attention = SpatialAttention(d_model)
        
        self.text_decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        print(f"TienetReportGenerator initialized with freeze_epochs={freeze_epochs}")

    def set_epoch(self, epoch):
        self.image_encoder.set_epoch(epoch)

    def forward(self, img, tgt_ids, tgt_mask=None, tgt_key_padding_mask=None):
        
        img_features = self.image_encoder(img)
        
        attended_features, spatial_attn_map = self.spatial_attention(img_features)
        
        B, C, H, W = attended_features.shape
        img_features_flat = attended_features.flatten(2).permute(0, 2, 1)  
        
        logits, decoder_attn_weights = self.text_decoder(
            tgt_ids, img_features_flat, tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        return logits, spatial_attn_map, decoder_attn_weights

    @torch.no_grad()
    def generate(self, img, tokenizer, max_len=128, beam_width=5,
                 start_token_id=None, end_token_id=None):
        
        self.eval()
        
        if start_token_id is None:
            if hasattr(tokenizer, 'cls_token_id') and tokenizer.cls_token_id is not None:
                start_token_id = tokenizer.cls_token_id
            elif hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
                start_token_id = tokenizer.bos_token_id
            else:
                start_token_id = tokenizer.pad_token_id  
        if end_token_id is None:
            if hasattr(tokenizer, 'sep_token_id') and tokenizer.sep_token_id is not None:
                end_token_id = tokenizer.sep_token_id
            elif hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
                end_token_id = tokenizer.eos_token_id
            else:
                end_token_id = tokenizer.pad_token_id  

        device = img.device
        
        img_features = self.image_encoder(img)
        attended_features, spatial_attn_map = self.spatial_attention(img_features)
        B, C, H, W = attended_features.shape
        img_features_flat = attended_features.flatten(2).permute(0, 2, 1) 
        
        initial_beam = (
            torch.full((1, 1), start_token_id, dtype=torch.long, device=device),
            0.0
        )
        beams = [initial_beam]
        completed_sequences = []
        
        for step in range(max_len - 1):
            if not beams:
                break
            
            new_beams = []
            possible_next_steps = []
            
            for seq_tensor, current_log_prob in beams:
                if seq_tensor[0, -1].item() == end_token_id:
                    score = current_log_prob / (seq_tensor.shape[1] ** 0.7)
                    completed_sequences.append((seq_tensor, score))
                    continue
                
                current_seq_len = seq_tensor.shape[1]
                tgt_mask = self.text_decoder.generate_square_subsequent_mask(current_seq_len).to(device)
                
                logits, _ = self.text_decoder(
                    seq_tensor, img_features_flat, tgt_mask=tgt_mask, tgt_key_padding_mask=None
                )
                
                next_token_logits = logits[:, -1, :]  # (1, vocab_size)
                next_token_log_probs = torch.log_softmax(next_token_logits, dim=-1)
                
                topk_log_probs, topk_indices = torch.topk(next_token_log_probs, beam_width, dim=-1)
                
                for i in range(beam_width):
                    next_token_id = topk_indices[0, i].item()
                    log_prob = topk_log_probs[0, i].item()
                    new_log_prob_total = current_log_prob + log_prob
                    possible_next_steps.append((seq_tensor, next_token_id, new_log_prob_total))
            
            possible_next_steps.sort(key=lambda x: x[2], reverse=True)
            
            added_beams = 0
            for seq_tensor, next_token_id, new_log_prob_total in possible_next_steps:
                if added_beams >= beam_width:
                    break
                
                new_seq_tensor = torch.cat([
                    seq_tensor,
                    torch.full((1, 1), next_token_id, dtype=torch.long, device=device)
                ], dim=1)
                
                is_duplicate = any(torch.equal(new_seq_tensor, b[0]) for b in new_beams)
                if not is_duplicate:
                    new_beams.append((new_seq_tensor, new_log_prob_total))
                    added_beams += 1
            
            beams = new_beams
        
        for seq_tensor, current_log_prob in beams:
            score = current_log_prob / (seq_tensor.shape[1] ** 0.7)
            completed_sequences.append((seq_tensor, score))
        
        completed_sequences.sort(key=lambda x: x[1], reverse=True)
        
        if completed_sequences:
            best_seq_tensor, _ = completed_sequences[0]
            generated_text = tokenizer.decode(best_seq_tensor[0], skip_special_tokens=True)
        else:
            generated_text = ""
        
        self.train()
        return generated_text, spatial_attn_map