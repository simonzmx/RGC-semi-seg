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


class SSEGCPS:

    def __init__(self, args, hyp, weight_unsup=None):
        self.device = 'cuda' if not args.cpu and torch.cuda.is_available() else 'cpu'
        self.model_save_path1 = Path(hyp['model_save_path'])
        self.model_save_path2 = Path(hyp['model_save_path2'])
        self.pretrained_path = Path(hyp['pretrained_path']) if hyp['pretrained_path'] is not None else None
        self.exp_name = args.exp_name if args.exp_name is not None else '.'.join(self.model_save_path1.name.split('.')[:-1])
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
        self.loss_cps = nn.CrossEntropyLoss()

        self.model1 = WeakGCSeg(hyp['num_channels'], self.num_classes)
        self.model1.to(self.device)
        self.model2 = WeakGCSeg(hyp['num_channels'], self.num_classes)
        self.model2.to(self.device)

        if self.pretrained_path and os.path.exists(self.pretrained_path):
            print('Loading encoder weights from the pre-trained model ...')
            pretrained_dict = torch.load(self.pretrained_path, map_location=torch.device(self.device))
            encoder_dict = {k: v for k, v in pretrained_dict.items() if 'Encoder.' in k}
            model_dict1 = self.model1.state_dict()
            model_dict1.update(encoder_dict)
            self.model1.load_state_dict(model_dict1)
            model_dict2 = self.model2.state_dict()
            model_dict2.update(encoder_dict)
            self.model2.load_state_dict(model_dict2)
            del pretrained_dict, encoder_dict, model_dict1, model_dict2
            # load full pretrained model
            # self.model.load_state_dict(torch.load(self.pretrained_path, map_location=torch.device(self.device)))

        if hyp['optimizer_name'].lower() == 'sgd':
            self.optimizer1 = torch.optim.SGD(self.model1.parameters(), lr=self.learning_rate,
                                              momentum=hyp['momentum'], weight_decay=hyp['weight_decay'])
            self.optimizer2 = torch.optim.SGD(self.model2.parameters(), lr=self.learning_rate,
                                              momentum=hyp['momentum'], weight_decay=hyp['weight_decay'])
        elif hyp['optimizer_name'].lower() == 'adam':
            self.optimizer1 = torch.optim.Adam(self.model1.parameters(), lr=self.learning_rate,
                                               weight_decay=hyp['weight_decay'])
            self.optimizer2 = torch.optim.Adam(self.model2.parameters(), lr=self.learning_rate,
                                               weight_decay=hyp['weight_decay'])
        else:
            self.optimizer1 = torch.optim.AdamW(self.model1.parameters(), lr=self.learning_rate,
                                                weight_decay=hyp['weight_decay'])
            self.optimizer2 = torch.optim.AdamW(self.model2.parameters(), lr=self.learning_rate,
                                                weight_decay=hyp['weight_decay'])
        # learning rate scheduler
        if hyp['lr_scheduler'] == 'poly':
            self.lr_scheduler1 = PolynomialLR(self.optimizer1, power=0.9,
                                              total_iters=self.epochs * math.ceil(
                                                  self.epoch_samples_train / self.batch_size))
            self.lr_scheduler2 = PolynomialLR(self.optimizer2, power=0.9,
                                              total_iters=self.epochs * math.ceil(
                                                  self.epoch_samples_train / self.batch_size))
        elif hyp['lr_scheduler'] == 'custom_poly':
            warmup_epochs = hyp['warmup_epochs'] \
                if 'warmup_epochs' in hyp and hyp['warmup_epochs'] is not None else int(self.epochs * 0.2)
            self.lr_scheduler1 = CustomPolynomialLR(self.optimizer1, power=0.9,
                                                    total_iters=self.epochs * math.ceil(
                                                        self.epoch_samples_train / self.batch_size),
                                                    warmup_epochs=warmup_epochs * math.ceil(
                                                        self.epoch_samples_train / self.batch_size))
            self.lr_scheduler2 = CustomPolynomialLR(self.optimizer1, power=0.9,
                                                    total_iters=self.epochs * math.ceil(
                                                        self.epoch_samples_train / self.batch_size),
                                                    warmup_epochs=warmup_epochs * math.ceil(
                                                        self.epoch_samples_train / self.batch_size))
        elif hyp['lr_scheduler'] == 'linear':
            lr_decay_epochs = hyp['lr_decay_epochs'] \
                if 'lr_decay_epochs' in hyp and hyp['lr_decay_epochs'] is not None else int(self.epochs * 0.5)
            end_factor = hyp['end_factor'] if 'end_factor' in hyp and hyp['end_factor'] is not None else 0.1
            self.lr_scheduler1 = LinearLR(self.optimizer1, start_factor=1.0, end_factor=end_factor,
                                          total_iters=lr_decay_epochs * math.ceil(
                                              self.epoch_samples_train / self.batch_size))
            self.lr_scheduler2 = LinearLR(self.optimizer2, start_factor=1.0, end_factor=end_factor,
                                          total_iters=lr_decay_epochs * math.ceil(
                                              self.epoch_samples_train / self.batch_size))
        else:
            self.lr_scheduler1 = None
            self.lr_scheduler2 = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        # tensorboard writer
        self.tfb_writer = SummaryWriter(os.path.join(self.results_path, 'tensorboard'))

        # Evaluator
        self.evaluator = Evaluator(args, hyp)

    def save_model_weights(self, model_index=1):
        if not os.path.exists(self.model_save_path1.parent):
            os.makedirs(self.model_save_path1.parent)
        if not os.path.exists(self.model_save_path2.parent):
            os.makedirs(self.model_save_path2.parent)

        if not os.path.exists(self.model_save_path1) or model_index == 1:
            torch.save(self.model1.state_dict(), self.model_save_path1)
            print('Model 1 saved.')
        if not os.path.exists(self.model_save_path2) or model_index == 2:
            torch.save(self.model2.state_dict(), self.model_save_path2)
            print('Model 2 saved.')

    def load_model_weights(self):
        if not os.path.exists(self.model_save_path1) and not os.path.exists(self.model_save_path2):
            raise FileNotFoundError('No trained models available.')
        if os.path.exists(self.model_save_path1):
            self.model1.load_state_dict(torch.load(self.model_save_path1))
            self.model1.to(self.device)
            print('Model 1 loaded.')
        if os.path.exists(self.model_save_path2):
            self.model2.load_state_dict(torch.load(self.model_save_path2))
            self.model2.to(self.device)
            print('Model 2 loaded.')

    def save_checkpoint(self, epoch, best_info):
        ckpt_path = os.path.join(self.results_path, 'last_ckpt.pt')
        state_dict = {
            'epoch': epoch,
            'model_state_dict1': self.model1.state_dict(),
            'optimizer_state_dict1': self.optimizer1.state_dict(),
            'scheduler_state_dict1': self.lr_scheduler1.state_dict() if self.lr_scheduler1 is not None else None,
            'model_state_dict2': self.model2.state_dict(),
            'optimizer_state_dict2': self.optimizer2.state_dict(),
            'scheduler_state_dict2': self.lr_scheduler2.state_dict() if self.lr_scheduler2 is not None else None,
            'best_info': best_info,
        }
        torch.save(state_dict, ckpt_path)
        print('Checkpoint saved.')

    def load_checkpoint(self) -> (int, dict):
        ckpt_path = os.path.join(self.results_path, 'last_ckpt.pt')
        if not os.path.exists(ckpt_path):
            print('No checkpoint available, training from scratch.')
            return 0, {'model_1': {'loss': math.inf, 'ap': 0, 'f1': 0, 'epoch': -1},
                       'model_2': {'loss': math.inf, 'ap': 0, 'f1': 0, 'epoch': -1},
                       'model_index': -1}
        else:
            state_dict = torch.load(ckpt_path)
            self.model1.load_state_dict(state_dict['model_state_dict1'])
            self.optimizer1.load_state_dict(state_dict['optimizer_state_dict1'])
            if self.lr_scheduler1 is not None:
                self.lr_scheduler1.load_state_dict(state_dict['scheduler_state_dict1'])
            self.model2.load_state_dict(state_dict['model_state_dict2'])
            self.optimizer2.load_state_dict(state_dict['optimizer_state_dict2'])
            if self.lr_scheduler2 is not None:
                self.lr_scheduler2.load_state_dict(state_dict['scheduler_state_dict2'])
            print('Checkpoint loaded {}.'.format(state_dict['best_info']))
            return state_dict['epoch'] + 1, state_dict['best_info']

    def _is_better(self, model_scores, best_model_info):
        if self.metric == 'ap':
            cond = True if model_scores[1] > best_model_info['ap'] else False
        elif self.metric == 'loss':
            cond = True if model_scores[0] < best_model_info['loss'] else False
        else:
            raise ValueError('Metric should be either \'ap\' or \'loss\'.')
        return cond

    def run_epoch(self, loader, epoch):
        self.model1.train()
        self.model2.train()
        loss_details = {'loss_ce_1': [], 'loss_ce_2': [],
                        'loss_cps_1': [], 'loss_cps_2': [],
                        'loss_total_1': [], 'loss_total_2': [],
                        'loss_total': []}
        progress_bar = tqdm.tqdm(loader)
        for curr_iter, batch_info in enumerate(progress_bar):
            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()
            if len(batch_info) == 5:
                batch_vols, batch_labels, batch_masks, batch_vols2, _ = batch_info
            elif len(batch_info) == 4:
                batch_vols, batch_labels, batch_vols2, _ = batch_info
                batch_masks = None
            else:
                raise ValueError(f"Batch information should have 4 or 5 elements, but got {len(batch_info)}")

            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.mixed_precision):
                batch_vols = batch_vols.to(self.device)
                batch_labels = batch_labels.to(self.device)
                if batch_masks is not None:
                    batch_masks = batch_masks.to(self.device)

                # supervised part
                logits_1 = self.model1(batch_vols)
                logits_2 = self.model2(batch_vols)

                if self.sparse_labels:
                    loss_ce_1 = self.loss_ce(logits_1, batch_labels, mask=batch_masks)
                    loss_ce_2 = self.loss_ce(logits_2, batch_labels, mask=batch_masks)
                else:
                    loss_ce_1 = self.loss_ce(logits_1, batch_labels)
                    loss_ce_2 = self.loss_ce(logits_2, batch_labels)

                if self.train_type == 'semi-supervised':
                    batch_vols2 = batch_vols2.to(self.device)
                    un_logits_1 = self.model1(batch_vols2)
                    un_logits_2 = self.model2(batch_vols2)
                    pred_1 = torch.cat([logits_1, un_logits_1], dim=0)  # use both batches in unsupervised training
                    pred_2 = torch.cat([logits_2, un_logits_2], dim=0)
                    _, max_1 = torch.max(pred_1, dim=1)
                    _, max_2 = torch.max(pred_2, dim=1)
                    max_1 = max_1.long()
                    max_2 = max_2.long()
                    loss_cps_1 = self.loss_cps(pred_1, max_2)
                    loss_cps_2 = self.loss_cps(pred_2, max_1)
                else:
                    loss_cps_1 = None
                    loss_cps_2 = None

            if logits_1.isnan().any() or logits_1.isinf().any():
                print(f'Nan/Inf values detected in the predicted supervised logits from model 1. epoch={epoch}')
                continue
            if logits_2.isnan().any() or logits_2.isinf().any():
                print(f'Nan/Inf values detected in the predicted supervised logits from model 2. epoch={epoch}')
                continue
            if un_logits_1.isnan().any() or un_logits_1.isinf().any():
                print(f'Nan/Inf values detected in the predicted unsupervised logits from model 1. epoch={epoch}')
                continue
            if un_logits_2.isnan().any() or un_logits_2.isinf().any():
                print(f'Nan/Inf values detected in the predicted unsupervised logits from model 2. epoch={epoch}')
                continue

            weight_cps = self.weight_unsup(epoch=epoch, curr_iter=curr_iter)
            loss_total_1 = loss_ce_1 + weight_cps * loss_cps_1 if loss_cps_1 else loss_ce_1
            loss_total_2 = loss_ce_2 + weight_cps * loss_cps_2 if loss_cps_2 else loss_ce_2
            loss_total = loss_total_1 + loss_total_2
            loss_details['loss_ce_1'].append(loss_ce_1.item())
            loss_details['loss_ce_2'].append(loss_ce_2.item())
            loss_details['loss_cps_1'].append(loss_cps_1.item())
            loss_details['loss_cps_2'].append(loss_cps_2.item())
            loss_details['loss_total_1'].append(loss_total_1.item())
            loss_details['loss_total_2'].append(loss_total_2.item())
            loss_details['loss_total'].append(loss_total.item())

            self.scaler.scale(loss_total).backward()
            self.scaler.unscale_(self.optimizer1)
            torch.nn.utils.clip_grad_norm_(self.model1.parameters(), self.max_norm)
            self.scaler.step(self.optimizer1)
            self.scaler.unscale_(self.optimizer2)
            torch.nn.utils.clip_grad_norm_(self.model2.parameters(), self.max_norm)
            self.scaler.step(self.optimizer2)
            self.scaler.update()

            progress_bar.set_description(
                ' '.join([f'{key}: {np.mean(values):.8f}' for key, values in loss_details.items()]) +
                f" GPU memory: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB" +
                f' lr: {self.lr_scheduler1.get_last_lr()[0]}'
            )

            # adjust learning rate
            if self.lr_scheduler1 is not None:
                self.lr_scheduler1.step()
            if self.lr_scheduler2 is not None:
                self.lr_scheduler2.step()

        self.tfb_writer.add_scalar(f'Loss_Total_1/train', np.mean(loss_details['loss_total_1']), epoch)
        self.tfb_writer.add_scalar(f'Loss_Total_2/train', np.mean(loss_details['loss_total_2']), epoch)
        self.tfb_writer.add_scalar(f'Loss_CE_1/train', np.mean(loss_details['loss_ce_1']), epoch)
        self.tfb_writer.add_scalar(f'Loss_CE_2/train', np.mean(loss_details['loss_ce_2']), epoch)
        self.tfb_writer.add_scalar(f'Loss_CPS_1/train', np.mean(loss_details['loss_cps_1']), epoch)
        self.tfb_writer.add_scalar(f'Loss_CPS_2/train', np.mean(loss_details['loss_cps_2']), epoch)
        self.tfb_writer.add_scalar(f'Learning_Rate', self.lr_scheduler1.get_last_lr()[0], epoch)
        print('Train: loss_1={:.4f}.'.format(np.mean(loss_details['loss_total_1'])))
        print('Train: loss_2={:.4f}.'.format(np.mean(loss_details['loss_total_2'])))

        return np.mean(loss_details['loss_total_1']), np.mean(loss_details['loss_total_2'])

    def train(self, train_loader, val_data_cfg=None):
        if self.resume_training:
            start_epoch, best_info = self.load_checkpoint()
        else:
            best_info = {'model_1': {'loss': math.inf, 'ap': 0, 'f1': 0, 'epoch': -1},
                         'model_2': {'loss': math.inf, 'ap': 0, 'f1': 0, 'epoch': -1},
                         'model_index': -1}
            start_epoch = 0
        torch.autograd.set_detect_anomaly(True)

        for epoch in range(start_epoch, self.epochs):
            print(f'---------- Epoch {epoch} ----------')
            self.run_epoch(train_loader, epoch)
            if val_data_cfg:
                # run validation
                scores_m1 = self.evaluator.evaluate(self.model1, self.loss_ce, val_data_cfg,
                                                    'val', self.tfb_writer, epoch=epoch, model_index=1)
                scores_m2 = self.evaluator.evaluate(self.model2, self.loss_ce, val_data_cfg,
                                                    'val', self.tfb_writer, epoch=epoch, model_index=2)
                if self._is_better(scores_m1, best_info['model_1']) or self._is_better(scores_m2, best_info['model_2']):
                    if self.metric == 'ap':
                        model_index = 1 if scores_m1[1] > scores_m2[1] else 2
                    else:
                        model_index = 1 if scores_m1[0] < scores_m2[0] else 2
                    best_info['model_index'] = model_index
                    if self._is_better(scores_m1, best_info['model_1']):
                        best_info['model_1'] = {'loss': scores_m1[0], 'ap': scores_m1[1], 'f1': scores_m1[2],
                                                'threshold': scores_m1[3], 'epoch': epoch}
                        self.save_model_weights(model_index=1)
                    if self._is_better(scores_m2, best_info['model_2']):
                        best_info['model_2'] = {'loss': scores_m2[0], 'ap': scores_m2[1], 'f1': scores_m2[2],
                                                'threshold': scores_m2[3], 'epoch': epoch}
                        self.save_model_weights(model_index=2)
                    with open(os.path.join(self.results_path, 'train_info.json'), 'w') as wf:
                        wf.write(json.dumps(best_info))

                # break if patience met
                if self.patience and epoch > best_info['model_1']['epoch'] + self.patience \
                        and epoch > best_info['model_2']['epoch'] + self.patience:
                    print(f'Validation performance has not been improved after {self.patience} epochs, training stopped.')
                    break
            else:
                self.save_model_weights()

            self.save_checkpoint(epoch, best_info)

        if val_data_cfg:
            print('Model 1:')
            print(best_info['model_1'])
            print('Model 2:')
            print(best_info['model_2'])
        else:
            print(f'No validation dataset provided, save the model weights after all {self.epochs} training epochs.')

        return best_info
