import os
import pandas as pd
import numpy as np


def save_scores_to_csv(scores, img_paths, results_path, model_index=None, train_info=None):
    keys = ['pre', 'rec', 'f1', 'ap', 'pre_3T', 'rec_3T', 'f1_3T', 'ap_3T',
            'pre_8T', 'rec_8T', 'f1_8T', 'ap_8T', 'pre_12T', 'rec_12T', 'f1_12T', 'ap_12T']

    scores_vt = {key: [] for key in keys}
    scores_tt = {key: [] for key in keys}
    for img_idx, img_path in enumerate(img_paths):
        img_filename = img_path.split(os.sep)[-1][:-4]
        this_dataset, this_subject, this_location = img_filename.split('_')[0: 3]
        if this_location == '13T':
            this_location = '12T'
        scores_vt['pre_{}'.format(this_location)].append(scores['pres_vt'][img_idx])
        scores_vt['rec_{}'.format(this_location)].append(scores['recs_vt'][img_idx])
        scores_vt['f1_{}'.format(this_location)].append(scores['f1s_vt'][img_idx])
        scores_vt['ap_{}'.format(this_location)].append(scores['avg_pres'][img_idx])
        scores_tt['pre_{}'.format(this_location)].append(scores['pres'][img_idx])
        scores_tt['rec_{}'.format(this_location)].append(scores['recs'][img_idx])
        scores_tt['f1_{}'.format(this_location)].append(scores['f1s'][img_idx])
        scores_tt['ap_{}'.format(this_location)].append(scores['avg_pres'][img_idx])

    info_vt = {key: '-' for key in keys}
    info_tt = {key: '-' for key in keys}
    for key in keys:
        if '_' in key:
            if len(scores_vt[key]) == 1:
                info_vt[key] = '{:.4f}'.format(scores_vt[key][0])
                info_tt[key] = '{:.4f}'.format(scores_tt[key][0])
            elif len(scores_vt[key]) > 1:
                info_vt[key] = '{:.4f}({:.4f})'.format(np.mean(scores_vt[key]), np.std(scores_vt[key]))
                info_tt[key] = '{:.4f}({:.4f})'.format(np.mean(scores_tt[key]), np.std(scores_tt[key]))

    info_vt['pre'] = '{:.4f}({:.4f})'.format(np.mean(scores['pres_vt']), np.std(scores['pres_vt']))
    info_vt['rec'] = '{:.4f}({:.4f})'.format(np.mean(scores['recs_vt']), np.std(scores['recs_vt']))
    info_vt['f1'] = '{:.4f}({:.4f})'.format(np.mean(scores['f1s_vt']), np.std(scores['f1s_vt']))
    info_vt['ap'] = '{:.4f}({:.4f})'.format(np.mean(scores['avg_pres']), np.std(scores['avg_pres']))

    info_tt['pre'] = '{:.4f}({:.4f})'.format(np.mean(scores['pres']), np.std(scores['pres']))
    info_tt['rec'] = '{:.4f}({:.4f})'.format(np.mean(scores['recs']), np.std(scores['recs']))
    info_tt['f1'] = '{:.4f}({:.4f})'.format(np.mean(scores['f1s']), np.std(scores['f1s']))
    info_tt['ap'] = '{:.4f}({:.4f})'.format(np.mean(scores['avg_pres']), np.std(scores['avg_pres']))

    if type(train_info) == dict:
        info_train = {f'val_{key}': train_info[key] for key in train_info.keys()}
        info_vt.update(info_train)
        info_tt.update(info_train)

    df_vt = pd.DataFrame(info_vt, index=[0])
    df_tt = pd.DataFrame(info_tt, index=[0])
    df = pd.concat([df_vt, df_tt], ignore_index=True)
    if model_index is None:
        df.to_csv(os.path.join(results_path, 'scores.csv'), index=False)
    else:
        df.to_csv(os.path.join(results_path, f'scores_{model_index}.csv'), index=False)
