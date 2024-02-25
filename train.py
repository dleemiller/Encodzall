import torch
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR, OneCycleLR
from torchmetrics import Accuracy

from encodzall import Tokenizer
from encodzall.config import load_s, load_train
from encodzall.model.autoencoder import AutoEncoder

from string_noise import noise


# init
train = load_train()
text_col = train.data.text_col
arch = load_s()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Tokenizer.init(arch)


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    a_norm = nn.functional.normalize(a, p=2, dim=1)
    b_norm = nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def build_dataloader():
    dataset = load_dataset(train.data.huggingface_id, streaming=train.data.streaming)
    trainset = dataset["train"].select_columns([text_col])

    # add noise
    trainset = trainset.map(
        lambda t: {
            "noise": noise.moe(
                t.get(text_col), probability=train.train.noise_probability
            )
        }
    )

    # tokenize
    trainset = trainset.map(lambda t: Tokenizer().tokenize(t.get("noise")))
    trainset = trainset.map(
        lambda t: {
            f"clean_{k}": v for k, v in Tokenizer().tokenize(t.get(text_col)).items()
        }
    )
    trainset = trainset.map(
        lambda t: Tokenizer().targets(
            input_ids=t["clean_input_ids"],
            attention_mask=t["clean_attention_mask"],
            word_start=t["clean_word_start"],
        )
    )

    # create dataloader
    return DataLoader(
        trainset,
        batch_size=train.train.batch_size,
        # shuffle=train.data.shuffle,
        num_workers=train.data.num_workers,
        pin_memory=train.data.pin_memory,
    )


def calculate_nll_loss(contrastive_loss, embeds0, embeds1):
    TEMPERATURE = train.train.temperature
    valid = torch.where(torch.any(embeds0 > 0, dim=1))[0]
    embeds0 = embeds0[valid]
    embeds1 = embeds1[valid]

    # SimCSE-style contrastive learning task for pooled embeddings
    labels = torch.arange(
        embeds0.shape[0],
        dtype=torch.long,
        device=embeds0.device,
        requires_grad=False,
    )
    csim = cos_sim(embeds0.squeeze(), embeds1.squeeze().detach()) * (1.0 / TEMPERATURE)
    return contrastive_loss(csim, labels)


model = AutoEncoder(arch).to(device)
model.train()
optimizer = AdamW(model.parameters(), lr=train.train.learning_rate)
sched = ExponentialLR(optimizer=optimizer, gamma=train.train.learning_rate_decay)
contrastive_loss = nn.CrossEntropyLoss().to(device)
dataloader = build_dataloader()
accuracy = Accuracy(
    task="multiclass",
    ignore_index=arch.tokenizer.pad_id,
    num_classes=arch.tokenizer.n_vocab,
).cuda()

for epoch in range(train.train.epochs):
    accum_loss = 0
    accum_loss_c = 0
    acc = 0
    for step, batch in enumerate(build_dataloader()):
        model.train()
        batch = {
            k: v.to(device) for k, v in batch.items() if k not in (text_col, "noise")
        }
        optimizer.zero_grad()
        reconstructive, embed0 = model(**batch)
        preds, target, loss = reconstructive
        if train.train.contrastive:
            with torch.no_grad():
                embed1 = model(
                    input_ids=batch["clean_input_ids"],
                    attention_mask=batch["clean_attention_mask"],
                    word_start=batch["clean_word_start"],
                    target_ids=None,
                    target_mask=None,
                    skip_decode=True,
                )
            loss_c = calculate_nll_loss(contrastive_loss, embed0, embed1)
            accum_loss_c += loss_c.item()
            (loss + loss_c).backward()
        else:
            loss.backward()
        optimizer.step()
        accum_loss += loss.item()
        acc += accuracy(preds, target)

        n = train.train.metrics_steps
        if step % n == n - 1:
            print(
                f"Loss ({epoch}:{step}): {accum_loss/n:.3f} contrastive={accum_loss_c/n:.3f}"
            )
            print(f"Accuracy ({epoch}:{step}):  {acc/n:.3f}")
            accum_loss = 0
            accum_loss_c = 0
            acc = 0
