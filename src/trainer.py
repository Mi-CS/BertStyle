import torch
from model import Similarity
from tqdm import tqdm
from torch.utils.data import DataLoader


def train_epoch(model,
            tokenizer,
            dataloader,
            optimizer,
            criterion,
            device,
            print_each, 
            disable_progress_bar): 

    """
    Trains the model using Contrastive Loss. 
    """

    sim = Similarity(temp = 0.5)
    num_batches = len(dataloader)

    max_tokens = 512
    running_loss = 0

    for i, batch in tqdm(enumerate(dataloader),
                         total=num_batches, 
                         disable=disable_progress_bar): 

        # Tokenize
        tok1 = tokenizer(batch[0], padding=True, return_tensors="pt")
        tok2 = tokenizer(batch[1], padding=True, return_tensors="pt")

        # Truncate if tensor size too big for BERT.
        tok1_input = tok1.input_ids[:, :max_tokens]
        tok1_att = tok1.attention_mask[:, :max_tokens]
        tok2_input = tok2.input_ids[:, :max_tokens]
        tok2_att = tok2.attention_mask[:, :max_tokens]        

        # To device
        tok1_input = tok1_input.to(device)
        tok1_att = tok1_att.to(device)
        tok2_input = tok2_input.to(device)
        tok2_att = tok2_att.to(device)

        # Gradients to zero
        optimizer.zero_grad()

        # Get representations BERT -> MLP -> out
        z1 = model(x_input = tok1_input,
                        x_att = tok1_att)
        z2 = model(x_input = tok2_input,
                        x_att = tok2_att)

        # Get cos similarities between both projections. 
        # Negative pairs of z1 are all in z2 but its
        # correspondent positive pair
        cos_sim = sim(z1.unsqueeze(1), z2.unsqueeze(0))

        # Build labels and compute contrastive loss.
        # Positive pairs lay in the diagonal
        # of cos_sim, so we can just assign them labels 
        # and use Cross Entropy. 
        labels = torch.arange(cos_sim.size(0)).long().to(device)
        loss = criterion(cos_sim, labels)

        # Backward pass + optimize
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        # Print train loss
        if (i+1) % print_each == 0: 
            print(f"Mean train loss for iterations {i+1-print_each} - {i+1}: {running_loss/print_each}")
            running_loss = 0
