"""
Base model and argument classes. All models should inherit these classes.
"""
from dataclasses import dataclass
import warnings

import torch
import torch.nn as nn
from transformers import (
    AdamW,
    AutoModel,
    AutoConfig,
    AutoModelForSequenceClassification,
)
import pytorch_lightning as pl
from typing import Union
from torch.utils.data import DataLoader


def get_text_model(
    model: Union[nn.Module, str], pretrained: bool, classifier: bool = False, **kwargs
) -> nn.Module:
    """Helper function for instantiating text models
    Args:
        model (nn.Module or str): the model to instantiate. If nn.Module, all other params ignored.
        pretrained (bool): whether to instantiate a pretrained-model
        classifier (bool): whether to add a classification head to the model
        **kwargs: goes to the from_pretrained() function for instance to pass num_labels for classifier
    """
    ModelMakerCls = AutoModelForSequenceClassification if classifier else AutoModel

    if isinstance(model, nn.Module):
        return model
    elif isinstance(model, str):
        if pretrained:
            return ModelMakerCls.from_pretrained(model, **kwargs)
        else:
            model_config = AutoConfig.from_pretrained(model, **kwargs)
            return ModelMakerCls.from_config(model_config)
    else:
        raise Exception("model must be a str or nn.Module")


@dataclass
class BaseModelArgs:
    """Arguments for BaseModel
    Args:
        lr (float): the learning rate for training the model
        adam_epsilon (float): ADAM epsilon parameter
    """

    lr: float = None
    adam_epsilon: float = None
    weight_decay: float = 0.0


@dataclass
class SemSupModelArgs(BaseModelArgs):
    """Arguments for SemSupModel
    Args:
        label_model (str or nn.Module): label_model or name of HF hub model
        pretrained_label_model (bool): If using a model from HF, load weights if set to True
        tune_label_model (bool): Whether the label model will be tuned throughout trianing
    """

    label_model: str = None
    pretrained_label_model: bool = True
    tune_label_model: bool = True

    def __post_init__(self):
        if not self.pretrained_label_model and not self.tune_label_model:
            warnings.warn("not tuning a scratch label model")


class BaseModel(pl.LightningModule):
    """Base class for all models. Inheriting models should override the forward() method and optionally define self.metrics"""

    def __init__(self, args: BaseModelArgs):
        super().__init__()
        self.args = args
        self.save_hyperparameters()

        # a dictionary, where key is the metric name and value is torchmetrics.Metric object
        self.metrics = dict()

    def forward(self, batch):
        """All inheriting classes should override this
        Args:
            batch: the batch from the input_loader

        Return (tuple): logits, targets, loss. Can be anything, as long as compatible with training_step(), validation_step()
            logits (Any): the logits generated by the model
            targets (Any): the targets
            loss (Any): the loss
        """
        raise NotImplementedError

    def step(self, batch):
        return self(batch)

    def configure_optimizers(self):
        return AdamW(
            self.parameters(),
            lr=self.args.lr,
            eps=self.args.adam_epsilon,
            weight_decay=self.args.weight_decay,
        )

    def training_step(self, batch, batch_idx):
        logits, targets, loss = self.step(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        logits, targets, loss = self.step(batch)
        for name, metric in self.metrics.items():
            metric(logits, targets)
        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("val_loss", loss, prog_bar=True)
        for name, metric in self.metrics.items():
            self.log(name, metric.compute(), prog_bar=True)
            metric.reset()
            
    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=BATCH_SIZE)


class SemSupModel(BaseModel):
    """Base class for all contclass models. Inheriting models should override the forward() method and optionally define self.metrics"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.label_model = get_text_model(
            model=self.args.label_model, pretrained=self.args.pretrained_label_model
        )

    def forward(self, batch, label_rep):
        """All inheriting classes should override this
        Args:
            batch: the batch from the input_loader
            label_rep (torch.Tensor[d_model, n_class]): torch tensor of the label representation from the label encoder

        Return (tuple): logits, targets, loss. Can be anything, as long as compatible with training_step(), validation_step()
            logits (Any): the logits generated by the model
            targets (Any): the targets
            loss (Any): the loss"""
      
        raise NotImplementedError

    def step(self, batch):
        label_batch = {k: v.squeeze(0) for k, v in batch["label_loader"].items()}
        with torch.set_grad_enabled(self.args.tune_label_model):
            # label_rep = self.label_model(
            #     **label_batch
            # ).pooler_output  # (n_class, d_model)
            
            output = self.label_model(**label_batch)
            hidden_state = output[0]  # (bs, seq_len, dim)
            label_rep = hidden_state[:, 0] # (bs, dim)
            
            label_rep = label_rep.t()  # (d_model, n_class)
        return self(batch["input_loader"], label_rep)
