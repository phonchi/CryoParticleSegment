"""Trainer."""
from collections import OrderedDict
import gc
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import reconstruct_patched, collate_fn
from metrics import ConfusionMatrix
from lr_scheduler import Callback
from utils import tqdm_plugin_for_loader
from plot import plot_result
from torch.optim.lr_scheduler import ReduceLROnPlateau
import segmentation_models_pytorch as smp

__all__ = ["Evaluator", "Trainer", "TrainerWithScheduler", "SemanticSegmentationTrainer",
           "CryoEMEvaluator", "CryoEMTrainer", "CryoEMTrainerWithScheduler", "tqdm_plugin_for_Trainer"]

class Evaluator():
    def __init__(self, model, device, metrics, num_classes: int = 2):
        self.model = model.to(device)
        self.device = device
        self.metrics = metrics
        self.num_classes = num_classes
        self.step_action = {}
    
    # def evaluate(self, loader, end_string: str=None):
        # self.model.eval()
        # self.initialize_evaluate(num_of_step=len(loader))
        # with torch.no_grad():
            # for batch_idx, (inputs, targets, *_) in enumerate(loader):
                # inputs = inputs.to(self.device)
                # targets = targets.to(self.device)
                # outputs = self.model(inputs)['out']
                # # Evaluating
                # self.step_evaluate(outputs, targets, batch_idx)
        # results = self.end_evaluate(end_string)
        # for func_name in self.step_action:
            # self.step_action[func_name](self, loader, results, batch_idx)
        # gc.collect()
        # torch.cuda.empty_cache()
        # return results
    def evaluate(self, loader, end_string: str=None):
        raise NotImplementedError
        
    def initialize_evaluate(self, num_of_step: int):
        self._results = OrderedDict()
    
    def step_evaluate(self, inputs, outputs, targets, batch_idx):
        raise NotImplementedError
    
    def end_evaluate(self, end_string: str=None) -> OrderedDict:
        if end_string is not None:
            print(end_string)
        return self._results
    
    def predict(self, loader):
        self.model.eval()
        predictions = list()
        with torch.no_grad():
            for batch_idx, (inputs, *_) in enumerate(loader):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)['out']
                predictions.extend(self.get_predictions(outputs).numpy())
        return predictions

    def get_predictions(self, outputs):
        preds = outputs.argmax(dim=1)
        return preds.cpu().detach()


class Trainer(Evaluator):
    """
    This trainer class is implemented only for four usages:
      Training, evaluation, prediction, and saving model.
    """
    def __init__(self, model, train_dataset, criterion, optimizer, device, 
                   metrics=['loss'], val_metrics=None, num_classes = 2, lr_scheduler=None):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.criterion = criterion
        self.optimizer = optimizer
        self.metrics = metrics
        self.val_metrics = metrics if val_metrics is None else val_metrics
        self.device = device
        self.best_epoch = 0
        self.num_classes = num_classes
        self.train_loss = list()
        self.loss = list()
        self.val_results = list()
        self.best_loss = np.inf
        self.step_action = {}
        self.lr_scheduler = lr_scheduler
    
    def train_per_epochs(self, loader, aux):
        self.model.train()
        results = OrderedDict(loss=np.zeros((len(loader))))
        if 'acc' in self.metrics:
            results['acc'] = np.zeros((len(loader)))
        for batch_idx, (inputs, targets, *_) in enumerate(loader):
            inputs = inputs.to(self.device)
            #print(targets.dtype)
            targets = targets.to(self.device)
            
            # Training
            self.optimizer.zero_grad()
            if aux:
                dic = self.model(inputs)
                main_loss = self.criterion(dic['out'], targets)
                aux_loss = self.criterion(dic['aux'], targets)
                loss = main_loss + 0.4 * aux_loss
            else:
                loss = self.criterion(self.model(inputs)['out'], targets)
                
            loss.backward()
            self.optimizer.step()
            
            if isinstance(self.lr_scheduler, ReduceLROnPlateau) or self.lr_scheduler==None:
                pass
            else:
                self.lr_scheduler.step()
            # Evaluating
            results['loss'][batch_idx] = loss.item()
            if 'acc' in self.metrics:
                pass
                #results['acc'][batch_idx] = self.get_accuracy(outputs, targets)
            for func_name in self.step_action:
                self.step_action[func_name](self, loader, results, batch_idx)
        del inputs, targets
        gc.collect()
        torch.cuda.empty_cache()
        return results
    
    def train(self, num_epochs, val_loader=None, batch_size=64, ckpt_dir=None, random_state=None, *, verbose=1, aux=False):
        # Set Loader
        gen = torch.Generator()
        gen.manual_seed(random_state)
        train_loader = DataLoader(self.train_dataset, batch_size, shuffle=True, generator=gen, pin_memory=True, collate_fn=collate_fn) 
        #train_loader = DataLoader(self.train_dataset, batch_size, shuffle=True, generator=gen, pin_memory=True)
        if val_loader is None:
            val_loader = DataLoader(self.train_dataset, batch_size, shuffle=False, pin_memory=True)
        
        # Training
        self._best_state = self.model.state_dict()
        for epoch in range(num_epochs):
            if verbose:
                print(f"Epoch {epoch + 1:3d}/{num_epochs:3d}:")
            # Training Step
            train_results = self.train_per_epochs(train_loader, aux)
            if verbose:
                # Log learning rate for each parameter group
                for i, group in enumerate(self.optimizer.param_groups):
                    print(f"Parameter Group {i}, Learning Rate: {group['lr']}")
            print("Training score:")
            for metric in self.metrics:
                print(f"  {metric}\t: {train_results[metric].mean():.4f}")
            self.train_loss.append(train_results['loss'].mean())
            # Evaluating Step
            gc.collect()
            torch.cuda.empty_cache()
            val_results = self.evaluate(val_loader, end_string="Validation score:")
            self.loss.append(loss := val_results['loss'].mean())
            self.val_results.append(val_results)
            # History
            if hasattr(self, 'callback'):
                stop_training = self.callback(epoch+1, loss, ckpt_dir, verbose=verbose)
                if stop_training:
                    break
            # Save best result
            if loss < self.best_loss:
                self.best_loss = loss
                self.best_epoch = epoch
                self._best_state = self.model.state_dict()
                if verbose:
                    print(f"Loss improve to {loss}.")
        self.model.load_state_dict(self._best_state)
        del(self._best_state)
    
    def save_model(self, ckpt_file, *, verbose=0, training=False):
        if ckpt_file is not None:
            torch.save(self.model.state_dict(), ckpt_file)
            if verbose:
                print(f"Saving model at {ckpt_file}")
        elif not training:
            if verbose:
                print("'ckpt_file' should be given.")
    
    def initialize_evaluate(self, num_of_step: int):
        self._results = OrderedDict()
        self._results['loss'] = np.zeros(num_of_step)
    
    def step_evaluate(self, outputs, targets, batch_idx):
        self._results['loss'][batch_idx] = self.criterion(outputs, targets).item()
    
    def end_evaluate(self, end_string: str=None) -> OrderedDict:
        results = super(Trainer, self).end_evaluate(end_string)
        print(f"  loss : {self._results['loss'].mean():.4f}")
        return results
    
    def get_accuracy(self, outputs, targets):
        preds = outputs.argmax(dim=1)
        corrects = preds.eq(targets.argmax(dim=1))
        return corrects.sum().item() / targets.size(0)


class TrainerWithScheduler(Trainer, Callback):
    def __init__(self, model, train_dataset, val_dataset, criterion, optimizer,
               device, metrics=['loss'], val_metrics=['loss'], num_classes = 2,
               lr_scheduler=None, patience=5):
        Trainer.__init__(self, model, train_dataset, criterion, optimizer,
                               device, val_dataset, metrics, val_metrics, num_classes, lr_scheduler)
        Callback.__init__(self, lr_scheduler, patience)


class SemanticSegmentationTrainer(Trainer):
    def __init__(self, *arg, **kwarg):
        super().__init__(*arg, **kwarg)
        self.plot = False
    
    def initialize_evaluate(self, num_of_step):
        super(SemanticSegmentationTrainer, self).initialize_evaluate(num_of_step)
        #device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tp_total = torch.zeros((self.num_classes,), dtype=torch.long, device='cpu')
        self.fp_total = torch.zeros((self.num_classes,), dtype=torch.long, device='cpu')
        self.fn_total = torch.zeros((self.num_classes,), dtype=torch.long, device='cpu')
        self.tn_total = torch.zeros((self.num_classes,), dtype=torch.long, device='cpu')

    def step_evaluate(self, outputs, targets, batch_idx):
        super(SemanticSegmentationTrainer, self).step_evaluate(outputs, targets, batch_idx)
        outputs = torch.softmax(outputs, dim=1)
        outputs = outputs.argmax(dim=1, keepdim=True)

        tp, fp, fn, tn = smp.metrics.get_stats(
            output=outputs,
            target=targets,
            mode='multiclass',
            num_classes=self.num_classes
        )
        # print("Device check before operation:")
        # print("tp device:", tp.device)
        # print("fp device:", fp.device)
        # print("fn device:", fn.device)
        # print("tn device:", tn.device)
        # print("self.tp_total device:", self.tp_total.device)
        # Summing over the batch dimension assuming it's the first dimension
        self.tp_total += tp.sum(0)
        self.fp_total += fp.sum(0)
        self.fn_total += fn.sum(0)
        self.tn_total += tn.sum(0)

    def end_evaluate(self, end_string: str=None)-> OrderedDict:
        result = super(SemanticSegmentationTrainer, self).end_evaluate(end_string)
        #if 'iou' in self.val_metrics:
        metrics = ['iou', 'precision', 'recall', 'accuracy', 'f1_score']
        metric_functions = {
            'iou': smp.metrics.iou_score,
            'precision': smp.metrics.precision,
            'recall': smp.metrics.recall,
            'accuracy': smp.metrics.accuracy,
            'f1_score': smp.metrics.f1_score
        }

        for metric in metrics:
            func = metric_functions[metric]
            score = func(self.tp_total, self.fp_total, self.fn_total, self.tn_total, reduction='none')
            result[metric] = score.numpy()  # Adjust as per actual deployment scenario
            print(f"{metric.capitalize()} by Class: {result[metric]}")
        return result

    # if 'confmat' in self._results.keys():
      # self._results['confmat'].reduce_from_all_processes()
    # result = super(SemanticSegmentationTrainer, self).end_evaluate(end_string)
    # if 'iou' in self.val_metrics:
      # result['iou'] = self._results['confmat'].get_row_iou()
      # print(self._results['confmat'])
    # return result


class CryoEMTrainer(SemanticSegmentationTrainer):
    def __init__(self, *arg, **kwarg):
        super().__init__(*arg, **kwarg)
    
    def evaluate(self, loader, end_string: str=None):
        self.model.eval()
        self.initialize_evaluate(num_of_step=len(loader))
        mini_batch_size = 9  # Number of patches to process at once
        with torch.no_grad():
            for idx, (inputs, targets, grid, mask) in enumerate(loader):
                num_batches = (inputs.size(0) + mini_batch_size - 1) // mini_batch_size
                patched_outputs = []
        
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * mini_batch_size
                    end_idx = min(start_idx + mini_batch_size, inputs.size(0))
                    patch_input = inputs[start_idx:end_idx].to(self.device)
                    output = self.model(patch_input)['out']
                    patched_outputs.append(output.cpu())  # Move to CPU to conserve GPU memory
        
                    del patch_input
                    torch.cuda.empty_cache()
        
                outputs = torch.cat(patched_outputs).to(self.device)
                del patched_outputs
        
                #targets = targets.to(self.device)
                #targets = reconstruct_patched(targets, grid)[None, :]
                mask = mask.to(self.device)
                mask = mask.unsqueeze(1)  # Adds a channel dimension
                outputs = reconstruct_patched(outputs, grid)[None, :]
                #print(targets)
                self.step_evaluate(outputs, mask.long(), idx)
                del outputs, targets
                torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        return self.end_evaluate(end_string)
    
    def predict(self, loader):
        self.model.eval()
        predictions = list()
        with torch.no_grad():
            for batch_idx, (inputs, targets, grid, *_) in enumerate(loader):
                # inputs = inputs.view(-1,*inputs.shape[-3:]).to(self.device)
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)['out']
                #inputs = reconstruct_patched(inputs, grid)
                outputs = reconstruct_patched(outputs, grid)
                preds = self.get_predictions(outputs)
                predictions.extend(preds.numpy())
        return predictions


class CryoEMEvaluator(Evaluator):
    def __init__(self, model, device, metrics, num_classes: int = 2, *arg, **kwarg):
        super().__init__(model, device, num_classes, *arg, **kwarg)
        self.plot = False

    def initialize_evaluate(self, num_of_step):
        super().initialize_evaluate(num_of_step)
        #device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tp_total = torch.zeros((self.num_classes,), dtype=torch.long, device='cpu')
        self.fp_total = torch.zeros((self.num_classes,), dtype=torch.long, device='cpu')
        self.fn_total = torch.zeros((self.num_classes,), dtype=torch.long, device='cpu')
        self.tn_total = torch.zeros((self.num_classes,), dtype=torch.long, device='cpu')

    def step_evaluate(self, outputs, targets, batch_idx):
        #super(CryoEMEvaluator, self).step_evaluate(outputs, targets, batch_idx)
        outputs = torch.softmax(outputs, dim=1)
        outputs = outputs.argmax(dim=1, keepdim=True)

        tp, fp, fn, tn = smp.metrics.get_stats(
            output=outputs,
            target=targets,
            mode='multiclass',
            num_classes=self.num_classes
        )
        # print("Device check before operation:")
        # print("tp device:", tp.device)
        # print("fp device:", fp.device)
        # print("fn device:", fn.device)
        # print("tn device:", tn.device)
        # print("self.tp_total device:", self.tp_total.device)
        # Summing over the batch dimension assuming it's the first dimension
        self.tp_total += tp.sum(0)
        self.fp_total += fp.sum(0)
        self.fn_total += fn.sum(0)
        self.tn_total += tn.sum(0)

    def end_evaluate(self, end_string: str=None)-> OrderedDict:
        result = super(CryoEMEvaluator, self).end_evaluate(end_string)
        #if 'iou' in self.val_metrics:
        metrics = ['iou', 'precision', 'recall', 'accuracy', 'f1_score']
        metric_functions = {
            'iou': smp.metrics.iou_score,
            'precision': smp.metrics.precision,
            'recall': smp.metrics.recall,
            'accuracy': smp.metrics.accuracy,
            'f1_score': smp.metrics.f1_score
        }

        for metric in metrics:
            func = metric_functions[metric]
            score = func(self.tp_total, self.fp_total, self.fn_total, self.tn_total, reduction='none')
            result[metric] = score.numpy()  # Adjust as per actual deployment scenario
            print(f"{metric.capitalize()} by Class: {result[metric]}")
        return result

    def evaluate(self, loader, end_string: str=None):
        self.model.eval()
        self.initialize_evaluate(num_of_step=len(loader))
        mini_batch_size = 27  # Number of patches to process at once
        with torch.no_grad():
            for idx, (inputs, targets, grid, mask) in enumerate(loader):
                num_batches = (inputs.size(0) + mini_batch_size - 1) // mini_batch_size
                print(inputs.shape)
                patched_outputs = []
        
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * mini_batch_size
                    end_idx = min(start_idx + mini_batch_size, inputs.size(0))
                    patch_input = inputs[start_idx:end_idx].to(self.device)
                    output = self.model(patch_input)['out']
                    patched_outputs.append(output.cpu())  # Move to CPU to conserve GPU memory
        
                    del patch_input
                    torch.cuda.empty_cache()
        
                outputs = torch.cat(patched_outputs).to(self.device)
                del patched_outputs
        
                #targets = targets.to(self.device)
                #targets = reconstruct_patched(targets, grid)[None, :]
                mask = mask.to(self.device)
                mask = mask.unsqueeze(1)  # Adds a channel dimension
                outputs = reconstruct_patched(outputs, grid)[None, :]
                #print(targets)
                self.step_evaluate(outputs, mask.long(), idx)
                del outputs, targets
                torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        return self.end_evaluate(end_string)

    def predict(self, loader):
        self.model.eval()
        predictions = list()
        with torch.no_grad():
            for batch_idx, (inputs, targets, grid, *_) in enumerate(loader):
                # inputs = inputs.view(-1,*inputs.shape[-3:]).to(self.device)
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)['out']
                #inputs = reconstruct_patched(inputs, grid)
                outputs = reconstruct_patched(outputs, grid)
                preds = self.get_predictions(outputs)
                predictions.extend(preds.numpy())
        return predictions

    # def initialize_evaluate(self, num_of_step):
        # self._results = OrderedDict()
        # self._results['confmat'] = ConfusionMatrix(self.num_classes, self.device)

    # def step_evaluate(self, outputs, targets, batch_idx):
        # self._results['confmat'].update(targets.argmax(dim=1).flatten(), outputs.argmax(dim=1).flatten())

    # def end_evaluate(self, end_string: str=None) -> OrderedDict:
        # self._results['confmat'].reduce_from_all_processes()
        # result = super(CryoEMEvaluator, self).end_evaluate(end_string)
        # if 'iou' in self.metrics:
          # result['iou'] = self._results['confmat'].get_row_iou()
          # print(self._results['confmat'])
        # return result

class CryoEMTrainerWithScheduler(CryoEMTrainer, Callback):
    def __init__(self, model, train_dataset, criterion, optimizer,
                 device, metrics=['loss'], val_metrics=['loss', 'iou'], num_classes = 2,
                 lr_scheduler=None, patience=10):
        CryoEMTrainer.__init__(self, model, train_dataset, criterion, optimizer,
                             device, metrics, val_metrics, num_classes, lr_scheduler)
        Callback.__init__(self, lr_scheduler, patience)


def tqdm_plugin_for_Trainer(trainer):
    """
    Present progress bar with tqdm for each epochs.
    
    Parameters
    ----------
    trainer : Trainer
      Class of a trainer.
    
    Returns
    -------
    TrainerWithTqdm : Trainer
      Class of a trainer.
    
    """
    class TrainerWithTqdm(trainer):
        def __init__(self, *arg, **kwarg):
            super().__init__(*arg, **kwarg)
        
        @tqdm_plugin_for_loader(desc="Training")
        def train_per_epochs(self, *arg, **kwarg):
            return super(TrainerWithTqdm, self).train_per_epochs(*arg, **kwarg)
        
        @tqdm_plugin_for_loader(desc="Validation")
        def evaluate(self, *arg, **kwarg):
            return super(TrainerWithTqdm, self).evaluate(*arg, **kwarg)
    
    return TrainerWithTqdm