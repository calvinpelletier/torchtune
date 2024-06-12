import torch
from fire import Fire
from pathlib import Path

from torchtune.models.llama3 import llama3_8b
from torchtune.datasets import alpaca_dataset
from torchtune.models.llama3 import llama3_tokenizer
from torchtune.modules.transformer import tmp


TOK_PATH = '/home/cpelletier/model/llama3-8b/original/tokenizer.model'
DEVICE = 'cuda'


class CLI:
    def __init__(
        self, 
        ds_path='~/slf/ds', 
        full_ft_model_path='~/out/full0/TODO.pth',
    ):
        self._ds_path = Path(ds_path).expanduser()
        self._full_ft_model_path = Path(full_ft_model_path).expanduser()

    def build_ds(self):
        model = llama3_8b()
        model.load_state_dict(torch.load(self._full_ft_model_path))
        model.layers[16].tmp = True
        model = model.to(DEVICE)
        model.eval()

        data = _setup_data()
        for i, batch in enumerate(data):
            tokens, labels = batch["tokens"], batch["labels"]
            mask = batch.get("mask", None)  # shape [b, s, s]
            input_pos = batch.get("input_pos", None)  # shape [b, s]
            tokens = tokens.to(DEVICE)
            num_tokens += tokens.numel()
            labels = labels.to(DEVICE)
            mask = mask.to(DEVICE) if mask is not None else None
            input_pos = input_pos.to(DEVICE) if input_pos is not None else None
            self._model(tokens, mask=mask, input_pos=input_pos)
            
            if i == 2:
                break
        
        print(len(tmp.x), len(tmp.y))
        for a, b in zip(tmp.x, tmp.y):
            print(a.shape, b.shape)


def _setup_data():
    tokenizer = llama3_tokenizer(path=TOK_PATH)
    ds = alpaca_dataset(tokenizer=tokenizer)
    sampler = DistributedSampler(
        ds,
        num_replicas=1,
        rank=0,
        shuffle=False,
        seed=0,
    )
    dataloader = DataLoader(
        dataset=ds,
        batch_size=32,
        sampler=sampler,
        collate_fn=partial(
            utils.padded_collate,
            padding_idx=tokenizer.pad_id,
            ignore_idx=torch.nn.CrossEntropyLoss().ignore_index,
        ),
    )
    return dataloader


if __name__ == '__main__':
    Fire(CLI)
