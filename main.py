from typing import List
import os
os.environ["WANDB_SILENT"] = "true"
import hydra
from omegaconf import OmegaConf
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import Logger as LightningLoggerBase
import torch

from src.utils import utils
from src.models.OneModel import OneModel
from src.models.MultiModels import MultiModels

import wandb

import time

import gc

from src.utils.calibration import temperature_scale_model
from pathlib import Path

seed = 13

# Change working directory to the script's directory
script_dir = Path(__file__).parent.absolute()
os.chdir(script_dir)

log = utils.get_logger(__name__)


# where to download the datasets
data_dir = os.path.join(script_dir, "data/")

# where to upload the weights and biases logs
my_project = "tutorial_notebook"
my_entity = "xyz"

# get device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


n_models = 1
alpha = 0


def average_checkpoints(checkpoint_paths: List[str]) -> dict:
    # Load all checkpoints
    checkpoints = [torch.load(path, map_location="cpu")
                   for path in checkpoint_paths]

    avg_dict = {}
    sum_sqr_dict = {}

    # Initialize using the first checkpoint
    for key, value in checkpoints[0].items():
        if isinstance(value, torch.Tensor):
            avg_dict[key] = value.clone()
            sum_sqr_dict[key] = (value.clone()) ** 2
        else:
            avg_dict[key] = value  # non-tensor, keep as is

    # Accumulate for remaining checkpoints
    for ckpt in checkpoints[1:]:
        for key in avg_dict.keys():
            value = ckpt[key]
            if isinstance(value, torch.Tensor):
                avg_dict[key] += value
                sum_sqr_dict[key] += value ** 2
            # For non-tensors, we assume they’re identical.

    # Compute the average and variance for tensor keys and add alpha-scaled variance
    for key in sum_sqr_dict:
        avg_dict[key] = avg_dict[key]/float(n_models)
        variance = sum_sqr_dict[key] / n_models - avg_dict[key] ** 2
        avg_dict[key] = avg_dict[key] + alpha * torch.sqrt(variance)

    return avg_dict


def rho(n_model, num_layers=18):
    gc.collect()

    if n_model == 0:
        model = "src.models.modules.cifar_model_zoo.mobilenetv2.MobileNetV2"
    else:
        model = "src.models.modules.resnet_cifar.ResNet" + str(num_layers)
    model = "src.models.modules.resnet_cifar.ResNet" + str(num_layers)

    config = {
        "trainer": {
            "_target_": "pytorch_lightning.Trainer",
            "accelerator": "gpu",
            "devices": 1,
            "min_epochs": 1,
            "max_epochs": 1,
            "enable_progress_bar": True,
        },
        "model": {
            "_target_": "src.models.OneModel.OneModel",
            "model": {
                "_target_": model
            },
        },
        "optimizer": {
            "_target_": "torch.optim.AdamW",
            "lr": 0.001
        },
        "datamodule": {
            "_target_": "src.datamodules.datamodules.CIFAR10DataModule",
            "data_dir": data_dir,
            "batch_size": 250,
            "num_workers": 10,
            "pin_memory": True,
            "shuffle": True,
            "trainset_data_aug": False,
            # This is the irreducible loss model training, so we train on the
            # holdout set (we call this set the "valset" in the global terminology for the dataset
            # splits). Thus, we need augmentation on the valset
            "valset_data_aug": True,
            "number_of_models": n_models,
            "model_number": n_model,
            # "trainset_corruption": {
            #    "label_noise": True,
            #    "input_noise": False,
            #    "structured_noise": False,
            #    "pc_corrupted": 0.1  # corrupt 10% of the training samples
            # }
        },
        "callbacks": {
            # We want to save that irreducible loss model with the lowest validation
            # loss (we validate on the "trainset", in global terminology for the
            # dataset splits).
            "model_checkpoint": {
                "_target_": "pytorch_lightning.callbacks.ModelCheckpoint",
                "monitor": "val_loss_epoch",
                "mode": "min",
                "save_top_k": 1,
                "save_last": True,
                "verbose": False,
                "dirpath": os.path.join("tutorial_outputs", f"irreducible_loss_model_{n_model}_{num_layers}"),
                "filename": "epoch_{epoch:03d}",
                "auto_insert_metric_name": False,
            },
        },
        "logger": {
            # Log with wandb, you could choose a different logger
            "wandb": {
                "_target_": "pytorch_lightning.loggers.wandb.WandbLogger",
                "project": my_project,
                "save_dir": ".",
                "entity": my_entity,
                "job_type": "train",
            }
        },
        "seed": seed,
        "debug": False,
        "ignore_warnings": True,
        "test_after_training": True,
        "base_outdir": "logs",
    }

    # convert config to OmegaConf structured dict (default for Hydra), and pretty-print
    config = OmegaConf.create(config)

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    wandb.init(settings=wandb.Settings(init_timeout=180, silent=True),
               name=f"HOLDOUT-{n_model}-" + str(num_layers))

    # Init lightning datamodule
    print(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(
        config.datamodule)
    datamodule.setup()

    # Init lightning model
    print(f"Instantiating model <{config.model._target_}>")
    # pl_model = OneModel.load_from_checkpoint("/home/dsi/ohaday/test/RHO/tutorial_outputs/irreducible_loss_model/epoch_004.ckpt")
    pl_model: LightningModule = hydra.utils.instantiate(
        config=config.model,
        optimizer_config=utils.mask_config(
            config.get("optimizer", None)
        ),  # When initialising the optimiser, you need to pass it the model parameters. As we haven't initialised the model yet, we cannot initialise the optimizer here. Thus, we need to pass-through the optimizer-config, to initialise it later. However, hydra.utils.instantiate will instatiate everything that looks like a config (if _recursive_==True, which is required here bc OneModel expects a model argument). Thus, we "mask" the optimizer config from hydra, by modifying the dict so that hydra no longer recognises it as a config.
        scheduler_config=utils.mask_config(
            config.get("scheduler", None)
        ),  # see line above
        datamodule=datamodule,
        _convert_="partial",
    )

    pl_model.to(device)

    # Init lightning callbacks. Here, we only use one callback: saving the model
    # with the lowest validation set loss.
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                print(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers. Here, we use wandb.
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                print(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    print(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    # Send config to all lightning loggers
    print("Logging hyperparameters!")
    trainer.logger.log_hyperparams(config)

    # Train the model.
    print("Starting training!")
    trainer.fit(
        pl_model,
        train_dataloaders=datamodule.val_dataloader(),
        val_dataloaders=datamodule.train_dataloader(),
    )

    """
    print("Calibrating model with temperature scaling...")

    # Calibrate the model using validation data
    temperature = temperature_scale_model(
        model=pl_model.model,
        # You can use val_dataloader if it's cleaner
        val_loader=datamodule.test_dataloader(),
        device=device
    )
    pl_model.temperature = temperature

    print(f"Optimal temperature: {temperature:.4f}")
    """

    # Evaluate model on test set, using the best model achieved during training
    if config.get("test_after_training") and not config.trainer.get("fast_dev_run"):
        print("Starting testing!")
        trainer.test(dataloaders=datamodule.test_dataloader())

    def evaluate_and_save_model_from_checkpoint_path(checkpoint_path, name):
        # load best model
        model = OneModel.load_from_checkpoint(checkpoint_path)

        # compute irreducible losses
        model.eval()
        irreducible_loss_and_checks = utils.compute_losses_with_sanity_checks(
            dataloader=datamodule.train_dataloader(), model=model, device=device
        )

        # save irred losses in same directory as model checkpoint
        path = os.path.join(
            os.path.dirname(trainer.checkpoint_callback.best_model_path),
            name,
        )
        torch.save(irreducible_loss_and_checks, path)

        return path

    saved_path = evaluate_and_save_model_from_checkpoint_path(
            trainer.checkpoint_callback.best_model_path, f"irred_losses_and_checks.pt"
    )

    print(f"Using monitor: {trainer.checkpoint_callback.monitor}")

    model_saved_path = trainer.checkpoint_callback.best_model_path

    # Print path to best checkpoint
    print(f"Best checkpoint path:\n{model_saved_path}")
    print(f"Best checkpoint irred_losses_path:\n{saved_path}")

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=pl_model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )
    
    wandb.finish()

    return saved_path, model_saved_path, datamodule


def model(model_name, saved_path, model_saved_path, avg=None, datamodule=None, num_layers=18):
    # You could choose any selection method here
    # ("src.curricula.selection_methods.xyz"). We have implemented:
    # reducible_loss_selection, uniform_selection, ce_loss_selection,
    # irreducible_loss_selection, gradnorm_ub_selection, and thers
    selection_method = model_name

    # Path to irreducible losses. Transferred from irreducible loss model training.
    # You can replace this with the path if you want to run target model training
    # without rerunning irreducible loss model training.
    path_to_irreducible_losses = saved_path

    config = {
        "model": {
            "_target_": "src.models.MultiModels.MultiModels",
            "large_model": {
                "_target_": "src.models.modules.resnet_cifar.ResNet" + str(num_layers)
            },
            "percent_train": 0.1,
        },
        "optimizer": {
            "_target_": "torch.optim.AdamW",
            "lr": 0.001
        },
        "trainer": {
            "_target_": "pytorch_lightning.Trainer",
            "accelerator": "gpu",
            "devices": 1,
            "min_epochs": 1,
            "max_epochs": 200,
            "enable_progress_bar": True,
        },
        "datamodule": {
            "_target_": "src.datamodules.datamodules.CIFAR10DataModule",
            "data_dir": data_dir,
            "batch_size": 200,
            "num_workers": 20,
            "pin_memory": True,
            "shuffle": True,
            "trainset_data_aug": True,
            "valset_data_aug": False,
            "cluster_sub_labels": (selection_method == "sublabels_reducible_loss_selection"),
        },
        "selection_method": {
            "_target_": "src.curricula.selection_methods." + selection_method
        },
        "callbacks": {
            "model_checkpoint": {
                "_target_": "pytorch_lightning.callbacks.ModelCheckpoint",
                "monitor": "val_acc_epoch",
                "mode": "max",
                "save_top_k": 1,
                "save_last": True,
                "verbose": False,
                "dirpath": os.path.join("tutorial_outputs", f"target_model_{num_layers}"), # make path job-specific
                "filename": "epoch_{epoch:03d}",
                "auto_insert_metric_name": False,
            },
        },
        "logger": {
            # Log with wandb, you could choose a different logger
            "wandb": {
                "_target_": "pytorch_lightning.loggers.wandb.WandbLogger",
                "project": my_project,
                "save_dir": ".",
                "entity": my_entity,
                "job_type": "train",
            }
        },
        "irreducible_loss_generator": {
            "_target_": "torch.load",
            "f": path_to_irreducible_losses,
        },
        "embedding_generator": {
            "_target_": "torch.load",
            "f": model_saved_path,
            "weights_only": False,
        },


        "debug": False,
        "ignore_warnings": True,
        "test_after_training": True,
        "seed": seed,
        "eval_set": "test",  # set to test if you want to evaluate on the test set
    }

    rho_config = {
        "model": {
            "_target_": "src.models.OneModel.OneModel",
            "model": {
                "_target_": "src.models.modules.resnet_cifar.ResNet18"
            },
        },
        "optimizer": {
            "_target_": "torch.optim.AdamW",
            "lr": 0.001
        },
        "seed": seed,
        "debug": False,
        "ignore_warnings": True,
        "test_after_training": True,
        "base_outdir": "logs",
    }

    # convert config to OmegaConf structured dict (default for Hydra), and print
    config = OmegaConf.create(config)
    rho_config = OmegaConf.create(rho_config)

    if (selection_method == "uniform_selection"):
        name = "UNIFORM-"
    else:
        name = "RHO-"
    name += str(num_layers)
    if avg is not None and avg != 0:
        name = "Z" + str(avg) + "W" + name

    wandb.init(settings=wandb.Settings(init_timeout=180), name=name)
    wandb.config.start_time = time.time()

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    if datamodule is None:
        # Init lightning datamodule
        print(f"Instantiating datamodule <{config.datamodule._target_}>")
        datamodule: LightningDataModule = hydra.utils.instantiate(
            config.datamodule)
        datamodule.setup()

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    # init irreducible loss generator (precomputed losses, or irreducible loss
    # model)
    irreducible_loss_generator = hydra.utils.instantiate(
        config.irreducible_loss_generator
    )
    embedding_generator = hydra.utils.instantiate(
        config.embedding_generator
    )
    # pl_model = MultiModels.load_from_checkpoint("C:/Users/ani/Desktop/RHO-Loss-main/tutorial_outputs/target_model/last-v11.ckpt", irreducible_loss_generator=irreducible_loss_generator, datamodule=datamodule)
    embedding_generator: LightningModule = hydra.utils.instantiate(
        config=rho_config.model,
        optimizer_config=utils.mask_config(
            config.get("optimizer", None)
        ),  # When initialising the optimiser, you need to pass it the model parameters. As we haven't initialised the model yet, we cannot initialise the optimizer here. Thus, we need to pass-through the optimizer-config, to initialise it later. However, hydra.utils.instantiate will instatiate everything that looks like a config (if _recursive_==True, which is required here bc OneModel expects a model argument). Thus, we "mask" the optimizer config from hydra, by modifying the dict so that hydra no longer recognises it as a config.
        scheduler_config=utils.mask_config(
            config.get("scheduler", None)
        ),  # see line above
        datamodule=datamodule,
        _convert_="partial",
    )

    embedding_generator.to(device)

    datamodule.add_embedding(embedding_generator)

    # pl_model = MultiModels.load_from_checkpoint("C:/Users/ani/Desktop/RHO-Loss-main/tutorial_outputs/target_model/last-v11.ckpt", irreducible_loss_generator=irreducible_loss_generator, datamodule=datamodule)

    # If precomputed losses are used, verify that the sorting
    # of the precomputes losses matches the dataset
    if type(irreducible_loss_generator) is dict:
        # instantiate a separate datamodule, so that the main datamodule is
        # instantiated with the same random seed whether or not the precomputed
        # losses are used
        datamodule_temp = hydra.utils.instantiate(config.datamodule)
        datamodule_temp.setup()
        utils.verify_correct_dataset_order(
            dataloader=datamodule_temp.train_dataloader(),
            sorted_target=irreducible_loss_generator["sorted_targets"],
            idx_of_control_images=irreducible_loss_generator["idx_of_control_images"],
            control_images=irreducible_loss_generator["control_images"],
            device=device,
            dont_compare_control_images=config.datamodule.get(
                "trainset_data_aug", False
            ),  # cannot compare images from irreducible loss model training run with those of the current run if there is trainset augmentation
        )

    del datamodule_temp
    irreducible_loss_generator = irreducible_loss_generator["irreducible_losses"]

    # init selection method
    print(
        f"Instantiating selection method <{config.selection_method._target_}>")
    selection_method = hydra.utils.instantiate(config.selection_method)

    # Init lightning model
    print(f"Instantiating models")
    # pl_model = MultiModels.load_from_checkpoint("C:/Users/ani/Desktop/RHO-Loss-main/tutorial_outputs/target_model/last-v11.ckpt", irreducible_loss_generator=irreducible_loss_generator, datamodule=datamodule)
    pl_model: LightningModule = hydra.utils.instantiate(
        config.model,
        selection_method=selection_method,
        irreducible_loss_generator=irreducible_loss_generator,
        embedding_generator=embedding_generator,
        datamodule=datamodule,
        optimizer_config=utils.mask_config(
            config.get("optimizer", None)
        ),  # When initialising the optimiser, you need to pass it the model parameters. As we haven't initialised the model yet, we cannot initialise the optimizer here. Thus, we need to pass-through the optimizer-config, to initialise it later. However, hydra.utils.instantiate will instatiate everything that looks like a config (if _recursive_==True, which is required here bc OneModel expects a model argument). Thus, we "mask" the optimizer config from hydra, by modifying the dict so that hydra no longer recognises it as a config.
        _convert_="partial",
        avg_weights=avg
    )

    pl_model.to(device)

    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                print(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    print(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    # Send config to all lightning loggers
    print("Logging hyperparameters!")
    trainer.logger.log_hyperparams(config)

    # create eval set
    if config.eval_set == "val":
        val_dataloader = datamodule.val_dataloader()
    elif config.eval_set == "test":
        val_dataloader = datamodule.test_dataloader()
        print(
            "Using the test set as the validation dataloader. This is for final figures in the paper"
        )

    # Train the model
    print("Starting training!")
    trainer.fit(
        pl_model,
        train_dataloaders=datamodule.train_dataloader(),
        # we pass the eval set as the validation set to trainer.fit because we want to know the eval set accuracy after each epoch
        val_dataloaders=val_dataloader,
    )

    # Evaluate model on test set, using the best model achieved during training
    if config.get("test_after_training") and not config.trainer.get("fast_dev_run"):
        print("Starting testing!")
        trainer.test(dataloaders=datamodule.test_dataloader())

    # Make sure everything closed properly
    print("Finalizing!")
    utils.finish(
        config=config,
        model=pl_model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Print path to best checkpoint
    print(
        f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    wandb.finish()


if __name__ == "__main__":
    saved_path = []
    for n_model in range(n_models):
        temp1, model_saved_path, datamodule = rho(n_model)
        saved_path.append(temp1)

    avg_state = average_checkpoints(saved_path)

    avg_path = os.path.join("tutorial_outputs", f"averaged_model.pt")
    torch.save(avg_state, avg_path)

    model("sublabels_reducible_loss_selection", avg_path, model_saved_path)
