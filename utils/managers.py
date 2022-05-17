import gc
import glob
import torch
import time
import os
import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from utils.display_utils import info


class ModelManager:
    """
    This class is used to keep tracking of model being trained. It is also used to save and load models.
    """

    def __init__(self, model_name, outputs_dir, progress_log_train=10, progress_log_val=10, patience=10):
        self.cur_epoch = 0
        self.batch_losses_train = []
        self.batch_losses_val = []
        self.last_updated = 0
        self.best_val_loss = float('Inf')
        self.outputs_dir = outputs_dir
        self.model_name = model_name
        self.progress_log_train = progress_log_train
        self.progress_log_val = progress_log_val
        self.patience = patience

    def __generate_chkpoint_name(self, base_name, epoch, best=False, type=None):
        """
        type: 'mdl' for model and 'mtc' for metric
        """
        if best:
            return f"{base_name}--Epoch{epoch:02d}--BEST--{type}.pkl"
        else:
            return f"{base_name}--Epoch{epoch:02d}--TMP--{type}.pkl"

    def __save_checkpoint(self, chkpt_name, model, valid_loss):
        """
        Saves the model weights with current validation loss
        """
        chkpt_path = os.path.join(self.outputs_dir, chkpt_name)
        torch.save({'model_state_dict': model.state_dict(), 'valid_loss': valid_loss}, chkpt_path)

    def load_checkpoint(self, chkpt_path, model):
        """
        Load model weights from file
        """
        state_dict = torch.load(os.path.join(self.outputs_dir, chkpt_path))
        model.load_state_dict(state_dict['model_state_dict'])
        return model, state_dict['valid_loss']

    @staticmethod
    def checkpoint_static_loader(chkpt_path, model):
        """
        Load model weights from file
        """
        state_dict = torch.load(chkpt_path)
        model.load_state_dict(state_dict['model_state_dict'])
        return model, state_dict['valid_loss']

    def __save_metrics(self, metrics_file):
        """
        Save train and validation losses to file
        :param metrics_file: File to save metrics to
        :return:
        """
        dev_dict = {'train_losses': self.batch_losses_train,
                    'val_losses': self.batch_losses_val}

        torch.save(dev_dict, os.path.join(self.outputs_dir, metrics_file))

    def load_metrics(self, metrics_file, load_locally=False):
        dev_dict = torch.load(os.path.join(self.outputs_dir, metrics_file))
        if load_locally is True:
            self.batch_losses_train = dev_dict['train_losses']
            self.batch_losses_val = dev_dict['val_losses']
        return dev_dict['train_losses'], dev_dict['val_losses']

    def update_train_loss(self, train_loss, step, total_steps, epoch_id, n_epochs):
        """
        Keep tracking of training loss.
        :param train_loss: Train loss for current batch
        :param step: Batch id
        :param total_steps: Total number of batches in train set
        :param epoch_id: Epoch number
        :param n_epochs: Total number of epochs
        :param print_freq: Print current loss every print_freq loss
        :return:
        """
        self.batch_losses_train.append(train_loss)
        if (step + 1) % self.progress_log_train == 0 or (step + 1) == total_steps:
            avg_loss = np.mean(self.batch_losses_train[-self.progress_log_train:])
            info(f"Training:   batch {step + 1:05d}/{total_steps:05d} [{int(100 * (step + 1) / total_steps)}%]"
                 f"from epoch {epoch_id:02d}/{n_epochs:02d} -- Loss = {avg_loss}")

    def update_val_loss(self, val_loss, step, total_steps, epoch_id, n_epochs):
        """
        Keep tracking of validation loss.
        :param val_loss: Validation loss for current batch
        :param step: Batch id
        :param total_steps: Total number of batches in validation set
        :param epoch_id: Epoch number
        :param n_epochs: Total number of epochs
        :param print_freq: Print current loss every print_freq loss
        :return:
        """
        self.cur_epoch = epoch_id
        self.batch_losses_val.append(val_loss)
        if (step + 1) % self.progress_log_val == 0 or (step + 1) == total_steps:
            # show avg loss of last batches instead of one batch loss
            avg_loss = np.mean(self.batch_losses_val[-self.progress_log_val:])
            info(f"Validating: batch {step + 1:05d}/{total_steps:05d} [{int(100 * (step + 1) / total_steps)}%]"
                 f"from epoch {epoch_id:02d}/{n_epochs:02d} -- Loss = {avg_loss}")

    def __purge_outpath(self, pattern):
        purged = glob.glob(os.path.join(self.outputs_dir, f'*{pattern}*'))
        # print(f"Glob purged: {purged}")
        purged = os.listdir(self.outputs_dir)
        purged = [os.path.join(self.outputs_dir, f) for f in purged if pattern in f]
        # print(f"os purged: {purged}")

        for filepath in purged:
            try:
                os.remove(filepath)
            except:
                print(f"Error deleting file @ {filepath}")

    def update_model(self, model, n_steps):
        """
        Save model to file if we got better results.
        :param n_steps:
        :param model: Current models
        :return: True if model was not updated for the last PATIENCE epochs, otherwise false
        """
        # Find avg performance in the last epoch
        epoch_losses = self.batch_losses_val[-n_steps:]
        epoch_val_loss = np.mean(epoch_losses)

        if epoch_val_loss < self.best_val_loss:
            # delete previous model from dir
            self.__purge_outpath(pattern='BEST')
            # update best result we get
            self.best_val_loss = epoch_val_loss
            # create model filename
            model_file_name = self.__generate_chkpoint_name(self.model_name, self.cur_epoch, best=True, type='model')
            self.__save_checkpoint(model_file_name, model, self.best_val_loss)
            metrics_file_name = self.__generate_chkpoint_name(self.model_name, self.cur_epoch, best=True,
                                                              type='metrics')
            self.__save_metrics(metrics_file_name)
            self.last_updated = 0
            info(f"Model updated, new loss {epoch_val_loss}")
        else:
            # delete previous model from dir
            self.__purge_outpath(pattern='TMP')
            model_file_name = self.__generate_chkpoint_name(self.model_name, self.cur_epoch, best=False, type='model')
            metrics_file_name = self.__generate_chkpoint_name(self.model_name, self.cur_epoch, best=False,
                                                              type='metrics')
            self.__save_checkpoint(model_file_name, model, epoch_val_loss)
            self.__save_metrics(metrics_file_name)

            self.last_updated += 1
            info(f"Current loss {epoch_val_loss} -- Best loss {self.best_val_loss}")

        return self.last_updated >= self.patience

    def reinitialize_manager(self):
        # make sure that data saved correctly (epoch number, losses, ...)
        # Should run this operation when manager loaded from the disk
        pass


class TrainManager:
    def __init__(self, n_epochs, loss_fn, optimizer, scheduler, device, model_manager, verbose=False,val_device='cuda:1') -> None:
        self.n_epochs = n_epochs
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.model_manager = model_manager
        self.verbose = verbose
        self.cur_epoch = 1
        self.device = device
        self.val_device = val_device

    def train_model(self, model, train_loader: DataLoader, val_loader: DataLoader, train_fun=None):
        self.start_epoch = self.cur_epoch
        model.to(self.device)
       
        for epoch_id in range(self.start_epoch, self.n_epochs + 1):
            if self.scheduler is not None:
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                for pg in self.optimizer.param_groups:  # assume we have one, edit code if used complicated structure
                    current_lr = pg['lr']
            info(f"Starting epoch {epoch_id}/{self.n_epochs} -- lr = {current_lr}")
            # train loop
            if train_fun is not None:
                model = train_fun(model)
            else:
                model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = model(data)
                if isinstance(self.loss_fn, nn.CrossEntropyLoss): # for multiclass classification 
                    loss = self.loss_fn(output.double(), target.long())
                elif isinstance(self.loss_fn, nn.BCEWithLogitsLoss): # for multilabel classification
                    loss = self.loss_fn(output.double(), target.double())
                else:
                    raise Exception("Unexpected loss function")
                loss.backward()
                self.optimizer.step()
                self.model_manager.update_train_loss(loss.item(), batch_idx, len(train_loader), epoch_id, self.n_epochs)
                
            # free memory
            data, target, loss = data.to('cpu'), target.to('cpu'), loss.to('cpu')
            del batch_idx, data, target, loss
            torch.cuda.empty_cache()
            gc.collect()
            # validation loop
            model.eval()
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = self.loss_fn(output, target)
                self.model_manager.update_val_loss(loss.item(), batch_idx, len(val_loader), epoch_id, self.n_epochs)
                
            data, target, loss = data.to('cpu'), target.to('cpu'), loss.to('cpu')
            del batch_idx, data, target, 
            torch.cuda.empty_cache()
            gc.collect()
            # update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            # track training process
            stop = self.model_manager.update_model(model, len(val_loader))
            if stop:
                info(
                    f"Training stopped after {self.cur_epoch} epochs becasue did not improved for the last {self.model_manager.patience} epochs")
                break
            self.cur_epoch += 1
        return model

    def val_model(self, model, val_loader: DataLoader):
        info("verifying model")
        total_steps = len(val_loader)
        self.start_epoch = self.cur_epoch
        model.to(self.device)
        # validation loop
        model.eval()
        avg_loss = 0
        for step, (data, target) in enumerate(val_loader):
            data, target = data.to(self.device), target.to(self.device)
            output = model(data)
            loss = self.loss_fn(output, target)

            if (step + 1) % 500 == 0 or (step + 1) == total_steps:
                info(
                    f"Validating: batch {step + 1:05d}/{total_steps:05d} [{int(100 * step / total_steps)}%]-- Loss = {loss}")

            avg_loss += loss.item()

        return avg_loss / total_steps

