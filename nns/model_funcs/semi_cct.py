import os
import math
import tqdm
import json
from pathlib import Path
from torch.optim.lr_scheduler import LinearLR, PolynomialLR
from nns.lr_schedulers import CustomPolynomialLR
from collections import OrderedDict
from nns.model_archs.cct import *
from torch.utils.tensorboard import SummaryWriter
from nns.losses import MaskedCELoss, softmax_mse_loss
from nns.evaluator import Evaluator


class CCT:

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
        self.num_channels = hyp['num_channels']
        self.num_classes = hyp['num_classes']
        self.mixed_precision = hyp['mixed_precision']

        self.learning_rate = hyp['learning_rate']
        self.max_norm = hyp['max_norm']
        self.epochs = hyp['num_epochs']
        self.epoch_samples_train = hyp['epoch_samples_train']
        self.patience = hyp['patience']
        self.metric = hyp['metric']

        self.perturbations = hyp['perturbations']
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
        self.loss_cst = softmax_mse_loss

        # Model
        self.encoder = Encoder(self.num_channels)
        self.decoder = Decoder(self.num_channels, self.num_classes)
        self.model = nn.Sequential(OrderedDict([
            ('Encoder', self.encoder),
            ('Decoder', self.decoder),
        ]))
        self.model.to(self.device)

        if self.pretrained_path and os.path.exists(self.pretrained_path):
            # print('Loading encoder weights from the pre-trained model ...')
            # pretrained_dict = torch.load(self.pretrained_path, map_location=torch.device(self.device))
            # encoder_dict = {k: v for k, v in pretrained_dict.items() if 'Encoder.' in k}
            # model_dict = self.model.state_dict()
            # model_dict.update(encoder_dict)
            # self.model.load_state_dict(model_dict)
            # del pretrained_dict, encoder_dict, model_dict
            print('Loading encoder and main decoder weights from the pre-trained model ...')
            if 'weak_gc_seg' in self.pretrained_path.__str__():
                pretrained_dict = torch.load(self.pretrained_path, map_location=torch.device(self.device))
                model_dict = self.model.state_dict()
                for k, v in model_dict.items():
                    model_dict[k] = pretrained_dict[k[8:]]
                self.model.load_state_dict(model_dict)
                del pretrained_dict, model_dict
            else:
                try:
                    self.model.load_state_dict(torch.load(self.pretrained_path, map_location=torch.device(self.device)))
                except Exception as ex:
                    print(ex)
                    raise RuntimeError('Improper pretrained model weights. Please use model trained by weak_gc_seg or v1.')

        if self.train_type == 'semi-supervised':
            self.aux_decoders = self.get_aux_decoders()
            self.aux_decoders.to(self.device)
        else:
            self.aux_decoders = []

        # optimizer
        num_decoders = len(self.aux_decoders) + 1
        trainable_params = [{'params': filter(lambda p: p.requires_grad, aux_d.parameters())} for aux_d in
                            self.aux_decoders]
        trainable_params.append({'params': filter(lambda p: p.requires_grad, self.decoder.parameters())})
        trainable_params.append({'params': filter(lambda p: p.requires_grad, self.encoder.parameters())})
                                 # 'lr': self.learning_rate / num_decoders})

        if hyp['optimizer_name'].lower() == 'sgd':
            self.optimizer = torch.optim.SGD(trainable_params, lr=self.learning_rate,
                                             momentum=hyp['momentum'], weight_decay=hyp['weight_decay'])
        elif hyp['optimizer_name'].lower() == 'adam':
            self.optimizer = torch.optim.Adam(trainable_params, lr=self.learning_rate,
                                              weight_decay=hyp['weight_decay'])
        else:
            self.optimizer = torch.optim.AdamW(trainable_params, lr=self.learning_rate,
                                               weight_decay=hyp['weight_decay'])

        model_params = sum([i.shape.numel() for i in list(self.encoder.parameters())]) \
                       + sum([i.shape.numel() for i in list(self.decoder.parameters())]) \
                       + sum([i.shape.numel() for aux_d in self.aux_decoders for i in list(aux_d.parameters())])
        opt_params = sum([i.shape.numel() for j in self.optimizer.param_groups for i in j['params']])
        assert opt_params == model_params, 'some params are missing in the opt'

        # learning rate scheduler
        if hyp['lr_scheduler'] == 'poly':
            self.lr_scheduler = PolynomialLR(self.optimizer, power=0.9,
                                             total_iters=self.epochs * math.ceil(
                                                 self.epoch_samples_train / self.batch_size))
        elif hyp['lr_scheduler'] == 'custom_poly':
            warmup_epochs = hyp['warmup_epochs'] \
                if 'warmup_epochs' in hyp and hyp['warmup_epochs'] is not None else int(self.epochs * 0.2)
            self.lr_scheduler = CustomPolynomialLR(self.optimizer, power=0.9,
                                                   total_iters=self.epochs * math.ceil(
                                                       self.epoch_samples_train / self.batch_size),
                                                   warmup_epochs=warmup_epochs * math.ceil(
                                                       self.epoch_samples_train / self.batch_size))
        elif hyp['lr_scheduler'] == 'linear':
            lr_decay_epochs = hyp['lr_decay_epochs'] \
                if 'lr_decay_epochs' in hyp and hyp['lr_decay_epochs'] is not None else int(self.epochs * 0.5)
            end_factor = hyp['end_factor'] if 'end_factor' in hyp and hyp['end_factor'] is not None else 0.1
            self.lr_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=end_factor,
                                         total_iters=lr_decay_epochs * math.ceil(
                                             self.epoch_samples_train / self.batch_size))
        else:
            self.lr_scheduler = None

        self.trainable_params = [p for d in trainable_params for p in d['params']]

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
            'aux_decoders_state_dict': [aux_d.state_dict() for aux_d in self.aux_decoders],
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
            for aux_idx in range(len(self.aux_decoders)):
                self.aux_decoders[aux_idx].load_state_dict(state_dict['aux_decoders_state_dict'][aux_idx])
            print('Checkpoint loaded {}.'.format(state_dict['best_info']))
            return state_dict['epoch'] + 1, state_dict['best_info']

    def get_aux_decoders(self):
        aux_decoders = []
        for perturbation in self.perturbations:
            if perturbation == "F-Noise":
                aux_decoders.append(FeatureNoiseDecoder(self.num_channels, self.num_classes))
            elif perturbation == "F-Drop":
                aux_decoders.append(FeatureDropDecoder(self.num_channels, self.num_classes))
            elif perturbation == "Obj-Mask":
                aux_decoders.append(ObjectMaskingDecoder(self.num_channels, self.num_classes))
            elif perturbation == "Con-Mask":
                aux_decoders.append(ContextMaskingDecoder(self.num_channels, self.num_classes))
            elif perturbation == "Spatial-Drop":
                aux_decoders.append(DropOutDecoder(self.num_channels, self.num_classes))
            else:
                raise ValueError(f"Unsupported auxiliary types: {perturbation}")
        return nn.ModuleList(aux_decoders)

    # def calculate_aux_decoder_results(self, in_features, out_main_decoder):
    #     # print(f"GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    #     out_features = []
    #     for aux_idx, aux_decoder in enumerate(self.aux_decoders):
    #         # apply this auxiliary decoder
    #         aux_decoder_result = aux_decoder(in_features, out_main_decoder)
    #         out_features.append(aux_decoder_result)
    #         # print(f"GPU memory: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")
    #     return out_features

    def calculate_consistency_loss(self, aux_logits, main_logits):
        assert len(aux_logits) > 0
        consistency_losses = [self.loss_cst(aux_logit, main_logits, use_softmax=True)
                              for aux_idx, aux_logit in enumerate(aux_logits)]
        loss_unsup = (sum(consistency_losses) / len(aux_logits))
        return loss_unsup, consistency_losses

    def run_epoch(self, loader, epoch):
        self.model.train()
        loss_details = {'loss_ce': [], 'loss_cst': [], 'loss_total': []}
        loss_cst_details = {f'loss_cst_{self.perturbations[aux_di]}': [] for aux_di in range(len(self.aux_decoders))}
        progress_bar = tqdm.tqdm(loader)
        for curr_iter, batch_info in enumerate(progress_bar):
            self.optimizer.zero_grad()
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
                logits = self.model(batch_vols)
                if self.sparse_labels:
                    loss_ce = self.loss_ce(logits, batch_labels, mask=batch_masks)
                else:
                    loss_ce = self.loss_ce(logits, batch_labels)
                loss_details['loss_ce'].append(loss_ce.item())

                # unsupervised part
                if self.train_type == 'semi-supervised':
                    batch_vols2 = batch_vols2.to(self.device)
                    feature_maps = self.encoder(batch_vols2)
                    main_logits = self.decoder(feature_maps).detach()
                    # aux_logits = []
                    # for aux_decoder in self.aux_decoders:
                    #     aux_logits.append(aux_decoder(feature_maps, main_logits))
                    #     print(f"GPU memory: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")
                    aux_logits = [aux_decoder(feature_maps, main_logits) for aux_decoder in self.aux_decoders]
                    loss_cst, cst_losses = self.calculate_consistency_loss(aux_logits, main_logits)
                    loss_details['loss_cst'].append(loss_cst.item())
                    for cst_idx, cst_loss in enumerate(cst_losses):
                        loss_cst_details[f'loss_cst_{self.perturbations[cst_idx]}'].append(cst_loss.item())
                else:
                    loss_cst = None

            weight_cst = self.weight_unsup(epoch=epoch, curr_iter=curr_iter)
            loss_total = loss_ce + weight_cst * loss_cst if loss_cst else loss_ce
            loss_details['loss_total'].append(loss_total.item())

            self.scaler.scale(loss_total).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.trainable_params, self.max_norm)
            # If these gradients do not contain ``inf``s or ``NaN``s, optimizer.step() is then called,
            # otherwise, optimizer.step() is skipped.
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.train_type == 'semi-supervised':
                del feature_maps
                del main_logits
                del aux_logits

            progress_bar.set_description(
                ' '.join([f'{key}: {np.mean(values):.8f}' for key, values in loss_details.items()]) +
                f' GPU memory: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB' +
                f' lr: {self.lr_scheduler.get_last_lr()[0]}'
            )

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()  # adjust learning rate

        self.tfb_writer.add_scalar(f'Loss_CE/train', np.mean(loss_details['loss_ce']), epoch)
        self.tfb_writer.add_scalar(f'Learning_Rate', self.lr_scheduler.get_last_lr()[0], epoch)
        if self.train_type == 'semi-supervised':
            self.tfb_writer.add_scalar(f'Loss_Consistency/train', np.mean(loss_details['loss_cst']), epoch)
            self.tfb_writer.add_scalar(f'Loss_Total/train', np.mean(loss_details['loss_total']), epoch)
            for ptb_idx, ptb in enumerate(self.perturbations):
                self.tfb_writer.add_scalar(f'Loss_cst_{ptb}/train', np.mean(loss_cst_details[f'loss_cst_{ptb}']), epoch)

        print('Train: loss={:.4f}.'.format(np.mean(loss_details['loss_total'])))

        return np.mean(loss_details['loss_total'])

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

            self.save_checkpoint(epoch, best_info)

        if val_data_cfg:
            print('Best epoch {}: loss={}, average_precision={}, f1={}, threshold={}'.format(
                best_info['epoch'], best_info['loss'], best_info['ap'], best_info['f1'], best_info['threshold']))
        else:
            print(f'No validation dataset provided, save the model weights after all {self.epochs} training epochs.')

        return best_info
