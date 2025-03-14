import os
import torch

from tqdm import tqdm

from src.constants import CHECKPOINT_DIR
from src.tracker import MetricTracker
from src.perplexity import Perplexity
from src.utils import dict_to_device
from transformers import Trainer
import json

class Trainer:
    def __init__(
        self,
        tokenizer,
        model,
        device,
        train_loader,
        valid_loader,
        optimizer,
        accum_grad_step,
        lr_scheduler,
        peft_type,
        #run_no,        
        logger=None,
        

        *arg,
        **kwarg,
        ):
        
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.train_num = len(train_loader)
        self.valid_num = len(valid_loader)
        self.optimizer = optimizer
        self.accum_grad_step = accum_grad_step
        self.lr_scheduler = lr_scheduler
        self.eval_func = Perplexity()
        self.tracker = MetricTracker()
        self.logger = logger
        self.peft_type = peft_type
        #self.run_no = run_no
        #print('save path')
        #print(self.peft_type)

    def train_step(self, batch_data, index):
        outputs = self.model(
            input_ids=batch_data["input_ids"],
            attention_mask=batch_data["attention_mask"],
            labels=batch_data["labels"],
        )
        loss = outputs.loss
        preds = outputs.logits.argmax(dim=-1)
        n = preds.shape[0]
        self.tracker.update("train/loss", loss / n, n)
        return loss

    def valid_step(self, batch_data, index):
        pred_logit = self.model(
            input_ids=batch_data["input_ids"],
            attention_mask=batch_data["attention_mask"],
        ).logits

        ppl = self.eval_func(
            pred_logits=pred_logit,
            labels=batch_data["input_ids"],
            output_masks=batch_data["output_mask"],
        )
        self.tracker.update(f"valid/ppl", ppl, pred_logit.shape[0])

        return

    def log(self, record):
        # self.progress_bar.set_postfix(record)
        if self.logger is not None:
            self.logger.log(record)
        return

    def train_one_epoch(self):
        self.model.train()
        self.progress_bar = tqdm(self.train_loader, desc=f"Training {self.cur_ep}")
        self.tracker.reset(keys=["train/loss"])

        for step, batch_data in enumerate(self.progress_bar, start=1):
            batch_data = dict_to_device(batch_data, self.device)
            loss = self.train_step(batch_data, step)
            self.progress_bar.set_postfix({**self.tracker.result(), "lr": self.lr_scheduler.get_last_lr()[0]})
            self.log({**self.tracker.result(), "lr": self.lr_scheduler.get_last_lr()[0]})

            (loss / self.accum_grad_step).backward()
            if step % self.accum_grad_step == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()

        self.progress_bar.close()
        return

    @torch.no_grad()
    def valid_one_epoch(self):
        self.model.eval()
        self.progress_bar = tqdm(self.valid_loader, desc=f"Validation {self.cur_ep}")
        self.tracker.reset(keys=["valid/ppl"])

        for step, batch_data in enumerate(self.progress_bar, start=1):
            batch_data = dict_to_device(batch_data, self.device)
            self.valid_step(batch_data, step)

        self.log({"epoch": self.cur_ep, **self.tracker.result()})
        self.progress_bar.close()
        return

    def fit(self, epoch):
        self.model.to(self.device)
        for self.cur_ep in range(1, epoch+1):
            self.train_one_epoch()
            self.valid_one_epoch()
            self.model.save_pretrained(
                os.path.join(
                CHECKPOINT_DIR,
                f"{self.peft_type}_{self.model.config.model_type}"  #  _{self.run_no}"
            )
        )
        pp = {"perplexity": self.tracker.result().get('valid/ppl', 0)}
        with open(f'checkpoint/{self.peft_type}_{self.model.config.model_type}/val_pp.json', 'w') as f:  
            json.dump(pp , f, indent=2) 
        return
