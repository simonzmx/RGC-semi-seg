import os
import json
from typing import Dict
from argparse import Namespace
from utils.datasets import GCSegmentationSparseDataset, GCSegmentationDataset
from torch.utils.data import DataLoader
from nns.weight_utils import consistency_weight
from nns.model_funcs.semi_cct import CCT
from nns.model_funcs.wgcs import WGCS
from nns.model_funcs.semi_cps import SSEGCPS
from nns.model_funcs.fixmatch import FixMatch


def process(main_args: Namespace,
            data_cfg: Dict,
            model_hyp: Dict):
    train_info = None
    if not main_args.no_train:
        if model_hyp['sparse_labels']:
            train_set = GCSegmentationSparseDataset(data_cfg, mode='train', dataset_type=model_hyp['train_type'])
        else:
            train_set = GCSegmentationDataset(data_cfg, mode='train', dataset_type=model_hyp['train_type'])
        train_dataloader = DataLoader(train_set, batch_size=main_args.batch_size, shuffle=True,
                                      pin_memory=True, num_workers=0)

        # # visualize batch data
        # import torch
        # for batch_vols, batch_labels, batch_vols_uw, batch_vols_us in train_dataloader:
        #     with torch.autocast(device_type='cuda', dtype=torch.float16):
        #         batch_vols = batch_vols.to('cuda').to(torch.float16)
        #         batch_labels = batch_labels.to('cuda').to(torch.float16)
        #         pass
        #         # print(f'bv: {batch_vols.min()}    {batch_vols.max()}')
        #         # print(f'bl: {batch_labels.min()}    {batch_labels.max()}')
        #         # print(f'bm: {batch_masks.min()}    {batch_masks.max()}')
        #         # print(f'bv2: {batch_vols_uw.min()}    {batch_vols_uw.max()}')
        #         # print(f'bv2: {batch_vols_us.min()}    {batch_vols_us.max()}')
        #         # assert not torch.any(batch_vols == 0)
        #         # assert not torch.any(batch_vols_uw == 0)
        #         # assert not torch.any(batch_vols_us == 0)

        # create model
        model = None
        if model_hyp['train_type'] == 'supervised':
            if model_hyp['model_name'] == 'WeakGCSeg':
                model = WGCS(main_args, model_hyp)
        elif model_hyp['train_type'] == 'semi-supervised':
            rampup_ends = int(model_hyp['ramp_up'] * model_hyp['num_epochs'])
            weight_unsup = consistency_weight(final_w=model_hyp['unsupervised_w'],
                                              iters_per_epoch=len(train_dataloader),
                                              rampup_ends=rampup_ends)
            if model_hyp['model_name'] == 'SemiSegCCT':
                model = CCT(main_args, model_hyp, weight_unsup=weight_unsup)
            elif model_hyp['model_name'] == 'SemiSegCPS':
                model = SSEGCPS(main_args, model_hyp, weight_unsup=weight_unsup)
            elif model_hyp['model_name'] == 'FixMatch':
                model = FixMatch(main_args, model_hyp, weight_unsup=weight_unsup)
        if model is None:
            raise ValueError("Unsupported training model/method.")

        train_info = model.train(train_dataloader, val_data_cfg=data_cfg)

    if not main_args.no_test:
        # create model
        model = None
        if model_hyp['model_name'] == 'WeakGCSeg':
            model = WGCS(main_args, model_hyp)
        elif model_hyp['model_name'] == 'SemiSegCCT':
            model = CCT(main_args, model_hyp)
        elif model_hyp['model_name'] == 'SemiSegCPS':
            model = SSEGCPS(main_args, model_hyp)
        elif model_hyp['model_name'] == 'FixMatch':
            model = FixMatch(main_args, model_hyp)
        if model is None:
            raise ValueError("Unsupported training model/method.")
        model.load_model_weights()

        # load training info
        if train_info is None:
            try:
                with open(os.path.join(model.results_path, 'train_info.json'), 'r') as rf:
                    train_info = json.loads(rf.read())
            except:
                pass

        # run evaluation
        if model_hyp['model_name'] == 'SemiSegCPS':
            if train_info is None:
                train_info_m1 = None
                train_info_m2 = None
            else:
                train_info_m1 = train_info['model_1']
                train_info_m2 = train_info['model_2']
            model.evaluator.evaluate(model.model1, model.loss_ce, data_cfg, 'test', model.tfb_writer,
                                     model_index=1, train_info=train_info_m1)
            model.evaluator.evaluate(model.model2, model.loss_ce, data_cfg, 'test', model.tfb_writer,
                                     model_index=2, train_info=train_info_m2)

            # # model ensemble by averaging two models' weights
            # sd1 = model.model1.state_dict()
            # sd2 = model.model2.state_dict()
            # for key in sd1:
            #     sd1[key] = (sd1[key] + sd2[key]) / 2.
            # model.model1.load_state_dict(sd1)
            # for key in train_info_m1:
            #     train_info_m1[key] = (train_info_m1[key] + train_info_m2[key]) / 2.
            # model.evaluator.evaluate(model.model1, model.loss_ce, data_cfg, 'test', model.tfb_writer,
            #                          model_index=3, train_info=train_info_m1)
        else:
            model.evaluator.evaluate(model.model, model.loss_ce, data_cfg, 'test', model.tfb_writer,
                                     train_info=train_info)

    return 0
