import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR, OneCycleLR
from torchmetrics import Accuracy

from encodzall import Tokenizer
from encodzall.config import load_s, load_train
from encodzall.model.autoencoder import AutoEncoder


# init
train = load_train()
arch = load_s()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Tokenizer.init(arch)


def build_dataloader():
    dataset = load_dataset(train.data.huggingface_id, streaming=train.data.streaming)
    trainset = dataset["train"].select_columns(["Text"])

    # tokenize
    trainset = trainset.map(lambda t: Tokenizer().tokenize(t.get("Text")))

    # add targets
    trainset = trainset.map(lambda t: Tokenizer().targets(**t))

    # create dataloader
    return DataLoader(
        trainset,
        batch_size=train.train.batch_size,
        # shuffle=train.data.shuffle,
        num_workers=train.data.num_workers,
        pin_memory=train.data.pin_memory,
    )


model = AutoEncoder(arch).to(device)
model.train()
optimizer = AdamW(model.parameters(), lr=train.train.learning_rate)
sched = ExponentialLR(optimizer=optimizer, gamma=train.train.learning_rate_decay)
dataloader = build_dataloader()
accuracy = Accuracy(
    task="multiclass",
    ignore_index=arch.tokenizer.pad_id,
    num_classes=arch.tokenizer.n_vocab,
).cuda()

for epoch in range(train.train.epochs):
    for step, batch in enumerate(build_dataloader()):
        batch = {k: v.to(device) for k, v in batch.items() if not k == "Text"}
        optimizer.zero_grad()
        preds, target, loss = model(**batch)
        print(loss)
        loss.backward()
        optimizer.step()

        if step % train.train.metrics_steps == 0:
            acc = accuracy(preds, target)
            print(f"Accuracy ({epoch}:{step})  {acc.item():.3f}")
