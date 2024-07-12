import os
import math
import torch
import tqdm
import json
import numpy as np
from pathlib import Path
from torch import nn
from torch.optim.lr_scheduler import LinearLR, PolynomialLR
from nns.lr_schedulers import CustomPolynomialLR
from nns.model_archs.wgcs import WeakGCSeg
from torch.utils.tensorboard import SummaryWriter
from nns.losses import MaskedCELoss
from nns.evaluator import Evaluator


class FixMatch:

    def __init__(self, args, hyp, weight_unsup=None):
        self.device = 'cuda' if not args.cpu and torch.cuda.is_available() else 'cpu'
        self.model_save_path = Path(hyp['model_save_path'])
        self.pretrained_path = Path(hyp['pretrained_path']) if hyp['pretrained_path'] is not None else None
        self.exp_name = args.exp_name if args.exp_name is not None else '.'.join(self.model_save_path.name.split('.')[:-1])
        self.results_path = os.path.join('runs', self.exp_name)
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)

        self.batch_size = args.batch_size
        self.resume_training = args.resume_training
        self.train_type = hyp['train_type']
        self.sparse_labels = hyp['sparse_labels']
        self.num_classes = hyp['num_classes']
        self.mixed_precision = hyp['mixed_precision']

        self.learning_rate = hyp['learning_rate']
        self.lr_scheduler = hyp['lr_scheduler']
        self.max_norm = hyp['max_norm']
        self.epochs = hyp['num_epochs']
        self.epoch_samples_train = hyp['epoch_samples_train']
        self.patience = hyp['patience']
        self.metric = hyp['metric']

        self.conf_thresh = hyp['conf_thresh']
        self.use_ema_model = hyp['ema_model']
        self.ema_start_epoch = hyp['ema_start_epoch']
        self.ema_decay = hyp['ema_decay']
        self.weight_unsup = weight_unsup

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

        # Unsupervised loss
        self.loss_unsup = nn.CrossEntropyLoss(reduction='none')

        # model
        self.model = WeakGCSeg(hyp['num_channels'], self.num_classes)
        self.model.to(self.device)

        if self.pretrained_path and os.path.exists(self.pretrained_path):
            print('Loading encoder weights from the pre-trained model ...')
            pretrained_dict = torch.load(self.pretrained_path, map_location=torch.device(self.device))
            encoder_dict = {k: v for k, v in pretrained_dict.items() if 'Encoder.' in k}
            model_dict = self.model.state_dict()
            model_dict.update(encoder_dict)
            self.model.load_state_dict(model_dict)
            del pretrained_dict, encoder_dict, model_dict
            # load full pretrained model
            # self.model.load_state_dict(torch.load(self.pretrained_path, map_location=torch.device(self.device)))

        # ema model
        if self.use_ema_model:
            self.model_ema = WeakGCSeg(hyp['num_channels'], self.num_classes)
            self.model_ema.load_state_dict(self.model.state_dict())
            for param in self.model_ema.parameters():
                param.detach_()
            self.model_ema.to(self.device)
        else:
            self.model_ema = self.model

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
            self.lr_scheduler = PolynomialLR(self.optimizer, power=0.9,
                                             total_iters=self.epochs * math.ceil(self.epoch_samples_train / self.batch_size))
        elif hyp['lr_scheduler'] == 'custom_poly':
            warmup_epochs = hyp['warmup_epochs'] \
                if 'warmup_epochs' in hyp and hyp['warmup_epochs'] is not None else int(self.epochs * 0.2)
            self.lr_scheduler = CustomPolynomialLR(self.optimizer, power=0.9,
                                                   total_iters=self.epochs * math.ceil(self.epoch_samples_train / self.batch_size),
                                                   warmup_epochs=warmup_epochs * math.ceil(self.epoch_samples_train / self.batch_size))
        elif hyp['lr_scheduler'] == 'linear':
            lr_decay_epochs = hyp['lr_decay_epochs'] \
                if 'lr_decay_epochs' in hyp and hyp['lr_decay_epochs'] is not None else int(self.epochs * 0.5)
            end_factor = hyp['end_factor'] if 'end_factor' in hyp and hyp['end_factor'] is not None else 0.1
            self.lr_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=end_factor,
                                         total_iters=lr_decay_epochs * math.ceil(self.epoch_samples_train / self.batch_size))
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
        torch.save(self.model_ema.state_dict(), self.model_save_path)
        print('Model saved.')

    def load_model_weights(self):
        if not os.path.exists(self.model_save_path):
            raise FileNotFoundError('No trained models available.')
        self.model_ema.load_state_dict(torch.load(self.model_save_path))
        self.model_ema.to(self.device)
        print('Model loaded.')

    def save_checkpoint(self, epoch, best_info):
        ckpt_path = os.path.join(self.results_path, 'last_ckpt.pt')
        state_dict = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            'model_state_dict_ema': self.model_ema.state_dict(),
            'best_info': best_info,
        }
        torch.save(state_dict, ckpt_path)
        print('Checkpoint saved.')

    def load_checkpoint(self) -> (int, dict):
        ckpt_path = os.path.join(self.results_path, 'last_ckpt.pt')
        if not os.path.exists(ckpt_path):
            print('No checkpoint available, training from scratch.')
            return 0, {'loss': math.inf, 'ap': 0, 'f1': 0, 'epoch': -1, 'model_index': -1}
        else:
            state_dict = torch.load(ckpt_path)
            self.model.load_state_dict(state_dict['model_state_dict'])
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            if self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(state_dict['scheduler_state_dict'])
            if self.use_ema_model:
                self.model_ema.load_state_dict(state_dict['model_state_dict_ema'])
            print('Checkpoint loaded {}.'.format(state_dict['best_info']))
            return state_dict['epoch'] + 1, state_dict['best_info']

    def _update_ema_variables(self, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        # for ema_param, param in zip(self.model_ema.parameters(), self.model.parameters()):
        #     ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
        for k, v in self.model_ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(alpha).add_(self.model.state_dict()[k], alpha=1 - alpha)

    # def check_models(self):
    #     models_differ = 0
    #     for p1, p2 in zip(self.model.parameters(), self.model.parameters()):
    #         if p1.data.ne(p2.data).sum() > 0:
    #             models_differ += 1
    #     if models_differ == 0:
    #         print('Model weights are the same!')
    #     else:
    #         print('Model weights are different!')
    #
    #     models_differ = 0
    #     for key_item_1, key_item_2 in zip(self.model.state_dict().items(), self.model_ema.state_dict().items()):
    #         if torch.equal(key_item_1[1], key_item_2[1]):
    #             pass
    #         else:
    #             models_differ += 1
    #             if key_item_1[0] == key_item_2[0]:
    #                 print(f'Mismtach found at {key_item_1[0]}')
    #             else:
    #                 raise Exception
    #     if models_differ == 0:
    #         print('Models match perfectly! :)')

    def run_epoch(self, loader, epoch):
        self.model.train()
        loss_details = {'loss_ce': [], 'loss_unsup': [], 'loss_total': [], 'conf_ratio': []}
        progress_bar = tqdm.tqdm(loader)
        n_batchs = len(loader)
        for curr_iter, batch_info in enumerate(progress_bar):
            self.optimizer.zero_grad()
            if len(batch_info) == 5:
                batch_vols, batch_labels, batch_masks, batch_vols_uw, batch_vols_us = batch_info
            elif len(batch_info) == 4:
                batch_vols, batch_labels, batch_vols_uw, batch_vols_us = batch_info
                batch_masks = None
            else:
                raise ValueError(f"Batch information should have 4 or 5 elements, but got {len(batch_info)}")

            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.mixed_precision):
                batch_vols = batch_vols.to(self.device)
                batch_labels = batch_labels.to(self.device)
                if batch_masks is not None:
                    batch_masks = batch_masks.to(self.device)

                # supervised part
                logits_sup = self.model(batch_vols)
                if self.sparse_labels:
                    loss_ce = self.loss_ce(logits_sup, batch_labels, mask=batch_masks)
                else:
                    loss_ce = self.loss_ce(logits_sup, batch_labels)

                # unsupervised
                batch_vols_uw = batch_vols_uw.to(self.device)
                batch_vols_us = batch_vols_us.to(self.device)
                # getting pseudo labels from the ema/teacher model
                with torch.no_grad():
                    self.model_ema.eval()
                    logits_uw = self.model_ema(batch_vols_uw).detach()
                    probs_uw = logits_uw.softmax(dim=1)
                    conf_uw, pseudo_label = probs_uw.max(dim=1)
                self.model.train()
                logits_us = self.model(batch_vols_us)
                loss_unsup = self.loss_unsup(logits_us, pseudo_label)
                # only use confident pseudo labels
                loss_unsup = loss_unsup * (conf_uw >= self.conf_thresh)
                loss_unsup = loss_unsup.sum() / batch_vols_us.numel()
                conf_ratio = (conf_uw >= self.conf_thresh).sum().item() / batch_vols_us.numel()

            weight_unsup = self.weight_unsup(epoch=epoch, curr_iter=curr_iter)
            loss_total = loss_ce + weight_unsup * loss_unsup if loss_unsup else loss_ce
            loss_details['loss_ce'].append(loss_ce.item())
            loss_details['loss_unsup'].append(loss_unsup.item())
            loss_details['loss_total'].append(loss_total.item())
            loss_details['conf_ratio'].append(conf_ratio)

            self.scaler.scale(loss_total).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.use_ema_model:
                ema_steps = max(0, (epoch - self.ema_start_epoch) * n_batchs + curr_iter)
                self._update_ema_variables(self.ema_decay, ema_steps)

            progress_bar.set_description(
                ' '.join([f'{key}: {np.mean(values):.8f}' for key, values in loss_details.items()]) +
                f' GPU memory: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB' +
                f' lr: {self.lr_scheduler.get_last_lr()[0]}'
            )

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()  # adjust learning rate

        self.tfb_writer.add_scalar(f'Loss_Total/train', np.mean(loss_details['loss_total']), epoch)
        self.tfb_writer.add_scalar(f'Loss_CE/train', np.mean(loss_details['loss_ce']), epoch)
        self.tfb_writer.add_scalar(f'Loss_Unsup/train', np.mean(loss_details['loss_unsup']), epoch)
        self.tfb_writer.add_scalar(f'Conf_Ratio/train', np.mean(loss_details['conf_ratio']), epoch)
        self.tfb_writer.add_scalar(f'Learning_Rate', self.lr_scheduler.get_last_lr()[0], epoch)
        print('Train: loss_={:.4f}.'.format(np.mean(loss_details['loss_total'])))

        return np.mean(loss_details['loss_total'])

    def train(self, train_loader, val_data_cfg=None):
        if self.resume_training:
            start_epoch, best_info = self.load_checkpoint()
        else:
            best_info = {'loss': math.inf, 'ap': 0, 'f1': 0, 'epoch': -1, 'model_index': -1}
            start_epoch = 0
        torch.autograd.set_detect_anomaly(True)

        for epoch in range(start_epoch, self.epochs):
            print(f'---------- Epoch {epoch} ----------')
            self.run_epoch(train_loader, epoch)
            if val_data_cfg:
                # run validation
                scores = self.evaluator.evaluate(self.model_ema, self.loss_ce, val_data_cfg,
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

            self.save_checkpoint(epoch, best_info)

        if val_data_cfg:
            print('Best epoch {}: loss={}, average_precision={}, f1={}, threshold={}'.format(
                best_info['epoch'], best_info['loss'], best_info['ap'],
                best_info['f1'], best_info['threshold']))
        else:
            print(f'No validation dataset provided, save the model weights after all {self.epochs} training epochs.')

        return best_info
