import os
import math
import torch
import tqdm
import json
import numpy as np
from pathlib import Path
from torch import nn
from torch.optim.lr_scheduler import LinearLR, PolynomialLR
from nns.model_archs.wgcs import WeakGCSeg
from nns.lr_schedulers import CustomPolynomialLR
from nns.losses import MaskedCELoss
from nns.evaluator import Evaluator
from torch.utils.tensorboard import SummaryWriter


class WGCS:

    def __init__(self, args, hyp):
        self.device = 'cuda' if not args.cpu and torch.cuda.is_available() else 'cpu'
        self.model_save_path = Path(hyp['model_save_path'])
        self.pretrained_path = Path(hyp['pretrained_path']) if hyp['pretrained_path'] is not None else None
        self.exp_name = args.exp_name if args.exp_name is not None else '.'.join(self.model_save_path.name.split('.')[:-1])
        self.results_path = os.path.join('runs', self.exp_name)
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)

        self.resume_training = args.resume_training
        self.train_type = hyp['train_type']
        self.sparse_labels = hyp['sparse_labels']
        self.num_classes = hyp['num_classes']
        self.mixed_precision = hyp['mixed_precision']

        self.learning_rate = hyp['learning_rate']
        self.max_norm = hyp['max_norm']
        self.epochs = hyp['num_epochs']
        self.patience = hyp['patience']
        self.metric = hyp['metric']

        # Supervised loss
        if hyp and 'loss_weights' in hyp and type(hyp['loss_weights']) == list:
            # get weights from model hyp file
            weight = torch.tensor(hyp['loss_weights']).to(self.device)
        else:
            weight = None
        if self.sparse_labels:
            self.loss_ce = MaskedCELoss(weight=weight)
        else:
            self.loss_ce = nn.CrossEntropyLoss(weight=weight)

        self.model = WeakGCSeg(hyp['num_channels'], self.num_classes)
        self.model.to(self.device)

        if self.pretrained_path and os.path.exists(self.pretrained_path):
            print('Loading weights from the pre-trained model ...')
            self.model.load_state_dict(torch.load(self.pretrained_path, map_location=torch.device(self.device)))

        if hyp['optimizer_name'].lower() == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,
                                             momentum=hyp['momentum'], weight_decay=hyp['weight_decay'])
        elif hyp['optimizer_name'].lower() == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                              weight_decay=hyp['weight_decay'])
        else:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate,
                                               weight_decay=hyp['weight_decay'])

        # learning rate scheduler
        if hyp['lr_scheduler'] == 'poly':
            self.lr_scheduler = PolynomialLR(self.optimizer, power=0.9, total_iters=self.epochs)
        elif hyp['lr_scheduler'] == 'custom_poly':
            self.lr_scheduler = CustomPolynomialLR(self.optimizer, power=0.9, total_iters=self.epochs,
                                                   warmup_epochs=int(self.epochs * 0.2))
        elif hyp['lr_scheduler'] == 'linear':
            total_iters = hyp['lr_decay_epochs'] \
                if 'lr_decay_epochs' in hyp and hyp['lr_decay_epochs'] is not None else int(self.epochs * 0.5)
            end_factor = hyp['end_factor'] if 'end_factor' in hyp and hyp['end_factor'] is not None else 0.1
            self.lr_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=end_factor, total_iters=total_iters)
        else:
            self.lr_scheduler = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        # tensorboard writer
        self.tfb_writer = SummaryWriter(os.path.join(self.results_path, 'tensorboard'))

        # Evaluator
        self.evaluator = Evaluator(args, hyp)

    def save_model_weights(self):
        if not os.path.exists(self.model_save_path.parent):
            os.makedirs(self.model_save_path.parent)
        torch.save(self.model.state_dict(), self.model_save_path)
        print('Model saved.')

    def load_model_weights(self):
        if not os.path.exists(self.model_save_path):
            raise FileNotFoundError('No trained model available.')
        self.model.load_state_dict(torch.load(self.model_save_path))
        self.model.to(self.device)
        print('Model loaded.')

    def save_checkpoint(self, epoch, best_info):
        ckpt_path = os.path.join(self.results_path, 'last_ckpt.pt')
        state_dict = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            'best_info': best_info,
        }
        torch.save(state_dict, ckpt_path)
        print('Checkpoint saved.')

    def load_checkpoint(self) -> (int, dict):
        ckpt_path = os.path.join(self.results_path, 'last_ckpt.pt')
        if not os.path.exists(ckpt_path):
            print('No checkpoint available, training from scratch.')
            return 0, {'loss': math.inf, 'ap': 0, 'f1': 0, 'epoch': -1}
        else:
            state_dict = torch.load(ckpt_path)
            self.model.load_state_dict(state_dict['model_state_dict'])
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            if self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(state_dict['scheduler_state_dict'])
            print('Checkpoint loaded {}.'.format(state_dict['best_info']))
            return state_dict['epoch'] + 1, state_dict['best_info']

    def run_epoch(self, loader, epoch):
        self.model.train()
        losses = []
        progress_bar = tqdm.tqdm(loader)
        for curr_iter, batch_info in enumerate(progress_bar):
            self.optimizer.zero_grad()
            if len(batch_info) == 5:
                batch_vols, batch_labels, batch_masks, _, _ = batch_info
            elif len(batch_info) == 4:
                batch_vols, batch_labels, _, _ = batch_info
                batch_masks = None
            else:
                raise ValueError(f"Batch information should have 4 or 5 elements, but got {len(batch_info)}")

            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.mixed_precision):
                batch_vols = batch_vols.to(self.device)
                batch_labels = batch_labels.to(self.device)
                if batch_masks is not None:
                    batch_masks = batch_masks.to(self.device)

                logits = self.model(batch_vols)
                if self.sparse_labels:
                    loss = self.loss_ce(logits, batch_labels, mask=batch_masks)
                else:
                    loss = self.loss_ce(logits, batch_labels)
                losses.append(loss.item())

            if logits.isnan().any() or logits.isinf().any():
                print(f'Nan/Inf values detected in the predicted supervised logits. epoch={epoch}')
                continue

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            progress_bar.set_description('loss: {:.8f}'.format(np.mean(losses)) +
                                         f' GPU memory: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB')

        self.tfb_writer.add_scalar(f'Loss/train', np.mean(losses), epoch)
        print('Train: loss={:.8f}.'.format(np.mean(losses)))

        return np.mean(losses)

    def train(self, train_loader, val_data_cfg=None):
        if self.resume_training:
            start_epoch, best_info = self.load_checkpoint()
        else:
            best_info = {'loss': math.inf, 'ap': 0, 'f1': 0, 'epoch': -1}
            start_epoch = 0
        torch.autograd.set_detect_anomaly(True)

        for epoch in range(start_epoch, self.epochs):
            print(f'---------- Epoch {epoch} ----------')
            self.run_epoch(train_loader, epoch)
            if val_data_cfg:
                # run validation
                scores = self.evaluator.evaluate(self.model, self.loss_ce, val_data_cfg,
                                                 'val', self.tfb_writer, epoch=epoch)
                if (self.metric == 'ap' and scores[1] > best_info['ap']) \
                        or (self.metric == 'loss' and scores[0] < best_info['loss']):
                    # store the best model
                    best_info = {'loss': scores[0], 'ap': scores[1], 'f1': scores[2],
                                 'threshold': scores[3], 'epoch': epoch}
                    self.save_model_weights()
                    with open(os.path.join(self.results_path, 'train_info.json'), 'w') as wf:
                        wf.write(json.dumps(best_info))
                else:
                    # break if patience met
                    if self.patience and epoch > best_info['epoch'] + self.patience:
                        print(f'Validation performance has not been improved after {self.patience} epochs, training stopped.')
                        break
            else:
                self.save_model_weights()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()  # adjust learning rate
                print(f'Learning rate adjusted to {self.lr_scheduler.get_last_lr()[0]}')

            self.save_checkpoint(epoch, best_info)

        if val_data_cfg:
            print('Best epoch {}: loss={}, average_precision={}, f1={}, threshold={}'.format(
                best_info['epoch'], best_info['loss'], best_info['ap'], best_info['f1'], best_info['threshold']))
        else:
            print(f'No validation dataset provided, save the model weights after all {self.epochs} training epochs.')
            self.save_model_weights()

        return best_info
