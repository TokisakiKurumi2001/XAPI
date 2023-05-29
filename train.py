from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from XAPI import XAPIDataLoader, LitXAPI

if __name__ == "__main__":
    wandb_logger = WandbLogger(project="proj_new_dummy")

    hyperparameter = {
        "ckpt": "xlm-roberta-base",
        "lr": 1e-3,
    }
    lit_xapi = LitXAPI(**hyperparameter)

    # dataloader
    xapi_pretrained_dataloader = XAPIDataLoader(ckpt=hyperparameter['ckpt'], max_length=128)
    [train_dataloader, test_dataloader, valid_dataloader] = xapi_pretrained_dataloader.get_dataloader(batch_size=128, types=["train", "test", "validation"])

    # train model
    trainer = pl.Trainer(max_epochs=2, devices=[0], accelerator="gpu", logger=wandb_logger)
    trainer.fit(model=lit_xapi, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
    trainer.test(dataloaders=test_dataloader)

    # save model & tokenizer
    lit_xapi.export_model('XAPI_model/v1')
