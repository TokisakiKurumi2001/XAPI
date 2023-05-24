from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from PAD import PADPDataLoader, LitPAD

if __name__ == "__main__":
    wandb_logger = WandbLogger(project="proj_new_dummy")

    hyperparameter = {
        "ckpt": "facebook/mbart-large-50",
        "lr": 3e-4,
    }
    lit_pad = LitPAD(**hyperparameter)

    # dataloader
    parableu_pretrained_dataloader = PADPDataLoader(ckpt=hyperparameter['ckpt'], max_length=128)
    [train_dataloader, test_dataloader, valid_dataloader] = parableu_pretrained_dataloader.get_dataloader(batch_size=64, types=["train", "test", "validation"])

    # train model
    trainer = pl.Trainer(max_epochs=2, devices=[0], accelerator="gpu", logger=wandb_logger, val_check_interval=5000)
    trainer.fit(model=lit_pad, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
    trainer.test(dataloaders=test_dataloader)

    # save model & tokenizer
    lit_pad.export_model('pad_model/v1')
