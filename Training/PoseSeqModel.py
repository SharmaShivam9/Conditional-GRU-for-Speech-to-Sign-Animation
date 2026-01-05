import torch
import torch.nn as nn
import random

class PoseSeqModel(nn.Module):
    """
    An Autoregressive Encoder-Decoder model for Text-to-Pose.
    
    This model has two "paths" in its forward method:
    
    1. TRAINING PATH (fast): 
       If a `target_sequence` is provided, it uses "full teacher forcing".
       It processes the *entire* sequence in one parallel pass, which is
       extremely fast on a GPU. It does *not* use a Python loop.
       
    2. INFERENCE PATH (slow):
       If `target_sequence` is None, it enters autoregressive mode.
       It uses a Python `for` loop to generate one frame at a time,
       feeding its own output back in as the input for the next step.
       This is required for generating new sequences.
    """
    def __init__(self, sentence_embed_dim, hidden_dim, pose_dim, n_gru_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pose_dim = pose_dim
        self.n_gru_layers = n_gru_layers
        self.sentence_embed_dim = sentence_embed_dim

        # 1. ENCODER
        self.encoder = nn.Linear(sentence_embed_dim, n_gru_layers * hidden_dim)
        self.encoder_activation = nn.ReLU()
        
        # 2. DECODER
        self.decoder_gru = nn.GRU(
            input_size=pose_dim + sentence_embed_dim, # pose + context
            hidden_size=hidden_dim,
            num_layers=n_gru_layers,
            batch_first=True,
            dropout=dropout if n_gru_layers > 1 else 0
        )
        
        # 3. OUTPUT LAYER
        self.output_linear = nn.Linear(hidden_dim, pose_dim)
        self.final_activation = nn.Sigmoid()

        # 4. START OF SEQUENCE (SOS) TOKEN
        self.sos_token = nn.Parameter(torch.randn(1, 1, pose_dim))

    def forward(self, sentence_embedding, Lmax, target_sequence=None, teacher_forcing_ratio=0.5):
        """
        Args:
            sentence_embedding (torch.Tensor): (Batch, sentence_embed_dim)
            Lmax (int): The max sequence length to unroll to.
            target_sequence (torch.Tensor, optional): 
                (Batch, Lmax, pose_dim) The "teacher" sequence. 
                If provided, the model runs in the fast "Training Path".
                If None, model runs in the slow "Inference Path".
        """
        B = sentence_embedding.shape[0]

        # 1. ENCODER Step (Common to both paths)
        # (B, Ein) -> (B, n_layers * H)
        h_encoded = self.encoder(sentence_embedding)
        h_encoded = self.encoder_activation(h_encoded)
        # (n_layers, B, H) - This is the initial hidden state h0
        hidden = h_encoded.view(B, self.n_gru_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        
        # We also need to re-feed the sentence embedding at each step.
        # (B, Ein) -> (B, 1, Ein)
        context = sentence_embedding.unsqueeze(1)
        
        decoder_input = self.sos_token.expand(B, 1, -1)

        outputs = []
        for t in range(Lmax):
            # 1. Combine last pose and context
            combined_input = torch.cat([decoder_input, context], dim=2)
            
            # 2. Run one step of the GRU
            gru_output, hidden = self.decoder_gru(combined_input, hidden)
            
            # 3. Get the new pose
            output_pose = self.output_linear(gru_output)
            output_pose = self.final_activation(output_pose)
            
            # 4. Store this frame's output
            outputs.append(output_pose)
            
            # 5. --- THIS IS YOUR NEW LOGIC ---
            # Decide what the *next* input will be
            
            use_teacher = False
            if target_sequence is not None:
                # If we are training, roll the dice FOR THIS FRAME
                if random.random() < teacher_forcing_ratio:
                    use_teacher = True
            
            if use_teacher:
                # Use the "teacher" (ground truth)
                decoder_input = target_sequence[:, t, :].unsqueeze(1) # (B, 1, pose_dim)
            else:
                # Use the model's own (detached) prediction
                decoder_input = output_pose.detach() # .detach() stops gradients
            
        # Combine all frames from the list
        all_outputs = torch.cat(outputs, dim=1)
            
        return all_outputs
    
