import json
import argparse

import numpy as np
import torch
from torch.distributions import kl_divergence
from torch.distributions.normal import Normal
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_rate', type=int, default=1)
    args = parser.parse_args()

    return args

def js_div(pro_mean, pro_std, anti_mean, anti_std):
    pro_dist = Normal(pro_mean, pro_std)
    anti_dist = Normal(anti_mean, anti_std)
    m_mean = (pro_mean + anti_mean) / 2
    m_std = (pro_std + anti_std) / 2
    m_dist = Normal(m_mean, m_std)

    pro_anti = kl_divergence(pro_dist, m_dist)
    anti_pro = kl_divergence(anti_dist, m_dist)
    jsd = (pro_anti + anti_pro) / 2
    return jsd


def my_kl_div(pro_mean, pro_std, anti_mean, anti_std):
    pro_dist = Normal(pro_mean, pro_std)
    anti_dist = Normal(anti_mean, anti_std)
    pro_anti = kl_divergence(pro_dist, anti_dist)
    anti_pro = kl_divergence(anti_dist, pro_dist)
    return pro_anti, anti_pro


def load_gms(path):
    file_name = path + '.json'
    output_result = path + '.txt'
    f = open(file_name)
    inputs = json.load(f)
    all_pro_mean_list = []
    all_pro_std_list = []
    all_anti_mean_list = []
    all_anti_std_list = []
    all_kl_score_list = []
    all_js_score_list = []
    all_len_list = []
    print(file_name)
    fw = open(output_result, 'w')
    for k, v in inputs.items():
        pro_list = v['pro']
        anti_list = v['anti']
        pro_list = torch.tensor(pro_list)
        anti_list = torch.tensor(anti_list)
        pro_std, pro_mean = torch.std_mean(pro_list)
        anti_std, anti_mean = torch.std_mean(anti_list)

        pro_anti, anti_pro = my_kl_div(pro_mean, pro_std, anti_mean, anti_std)
        score = torch.max(pro_anti / (pro_anti + anti_pro), anti_pro / (pro_anti + anti_pro)).item()
        js_score = js_div(pro_mean, pro_std, anti_mean, anti_std)
        KLDivS = score * 100
        JSDivS = (1 - js_score.item()) * 100
        fw.write(f'{k} KL: {round(KLDivS, 2)}\n')
        fw.write(f'{k} JS: {round(JSDivS, 2)}\n')

        all_pro_mean_list.append(pro_mean)
        all_pro_std_list.append(pro_std)
        all_anti_mean_list.append(anti_mean)
        all_anti_std_list.append(anti_std)
        all_kl_score_list.append(KLDivS)
        all_js_score_list.append(JSDivS)
        all_len_list.append(len(pro_list))
    weights = all_len_list / np.sum(all_len_list)
    mix_kl = 0.0
    mix_js = 0.0
    for kl, js, w in zip(all_kl_score_list, all_js_score_list, weights):
        mix_kl += w * kl
        mix_js += w * js
    print('Bias score KL:', round(mix_kl, 2))
    print('Bias score JS:', round(mix_js, 2))
    fw.write(f'Bias score KL: {round(mix_kl, 2)}\n')
    fw.write(f'Bias score JS: {round(mix_js, 2)}\n')
    return round(mix_kl, 2), round(mix_js, 2)


def load_others(path):
    file_name = path + '.json'
    output_result = path + '.txt'
    f = open(file_name)
    inputs = json.load(f)
    from collections import defaultdict
    count = defaultdict(int)
    scores = defaultdict(int)
    total_score = 0
    stereo_score = 0
    all_ranks = []
    for k, v in inputs.items():
        pro_list = v['pro']
        anti_list = v['anti']
        for pro_score, anti_score in zip(pro_list, anti_list):
            if pro_score > anti_score:
                stereo_score += 1
                scores[k] += 1
        count[k] = len(pro_list)
        total_score += len(pro_list)

        pro_ranks_list = v['pro_ranks']
        anti_ranks_list = v['anti_ranks']
        all_ranks += pro_ranks_list
        all_ranks += anti_ranks_list
    fw = open(output_result, 'w')
    all_bias_score = round((stereo_score / total_score) * 100, 2)
    print('Bias score:', all_bias_score)
    fw.write(f'Bias score: {all_bias_score}\n')
    for bias_type, score in sorted(scores.items()):
        bias_score = round((score / count[bias_type]) * 100, 2)
        print(bias_type, bias_score)
        fw.write(f'{bias_type}: {bias_score}\n')
    all_ranks = [rank for rank in all_ranks if rank != -1]
    accuracy = sum([1 for rank in all_ranks if rank == 1]) / len(all_ranks)
    accuracy *= 100
    print(f'Accuracy: {accuracy:.2f}')
    fw.write(f'Accuracy: {accuracy:.2f}\n')
    return all_bias_score


def load_jsdivs(path):
    file_name = path.replace('jsdivs', 'gms') + '.json'
    output_result = path + '.txt'
    f = open(file_name)
    inputs = json.load(f)
    all_pro_mean_list = []
    all_pro_std_list = []
    all_anti_mean_list = []
    all_anti_std_list = []
    all_js_score_list = []
    all_len_list = []
    print(file_name)
    fw = open(output_result, 'w')
    for k, v in inputs.items():
        pro_list = v['pro']
        anti_list = v['anti']
        pro_list = torch.tensor(pro_list)
        anti_list = torch.tensor(anti_list)
        pro_std, pro_mean = torch.std_mean(pro_list)
        anti_std, anti_mean = torch.std_mean(anti_list)

        pro_anti, anti_pro = my_kl_div(pro_mean, pro_std, anti_mean, anti_std)
        js_score = js_div(pro_mean, pro_std, anti_mean, anti_std)
        JSDivS = (1 - js_score.item()) * 100
        fw.write(f'{k} JS: {round(JSDivS, 2)}\n')

        all_pro_mean_list.append(pro_mean)
        all_pro_std_list.append(pro_std)
        all_anti_mean_list.append(anti_mean)
        all_anti_std_list.append(anti_std)
        all_js_score_list.append(JSDivS)
        all_len_list.append(len(pro_list))
    weights = all_len_list / np.sum(all_len_list)
    mix_js = 0.0
    for js, w in zip(all_js_score_list, weights):
        mix_js += w * js
    print('Bias score JS:', round(mix_js, 2))
    fw.write(f'Bias score JS: {round(mix_js, 2)}\n')
    return round(mix_js, 2)


def load_kldivs(path):
    file_name = path.replace('kldivs', 'gms') + '.json'
    output_result = path + '.txt'
    f = open(file_name)
    inputs = json.load(f)
    all_pro_mean_list = []
    all_pro_std_list = []
    all_anti_mean_list = []
    all_anti_std_list = []
    all_kl_score_list = []
    all_len_list = []
    print(file_name)
    fw = open(output_result, 'w')
    for k, v in inputs.items():
        pro_list = v['pro']
        anti_list = v['anti']
        pro_list = torch.tensor(pro_list)
        anti_list = torch.tensor(anti_list)
        pro_std, pro_mean = torch.std_mean(pro_list)
        anti_std, anti_mean = torch.std_mean(anti_list)

        pro_anti, anti_pro = my_kl_div(pro_mean, pro_std, anti_mean, anti_std)
        score = torch.max(pro_anti / (pro_anti + anti_pro), anti_pro / (pro_anti + anti_pro)).item()
        KLDivS = score * 100
        fw.write(f'{k} KL: {round(KLDivS, 2)}\n')

        all_pro_mean_list.append(pro_mean)
        all_pro_std_list.append(pro_std)
        all_anti_mean_list.append(anti_mean)
        all_anti_std_list.append(anti_std)
        all_kl_score_list.append(KLDivS)
        all_len_list.append(len(pro_list))
    weights = all_len_list / np.sum(all_len_list)
    mix_kl = 0.0
    for kl, w in zip(all_kl_score_list, weights):
        mix_kl += w * kl
    print('Bias score KL:', round(mix_kl, 2))
    fw.write(f'Bias score KL: {round(mix_kl, 2)}\n')
    return round(mix_kl, 2)


def scoreing(data, method, model, sample_rate):
    if sample_rate == 1:
        path = '../result/output/' + data + '_' + method + '_' + model
    else:
        path = '../result/output/' + str(sample_rate) + '_' + data + '_' + method + '_' + model
    # print(path)
    if method == 'kldivs':
        score = load_kldivs(path)
    elif method == 'jsdivs':
        score = load_jsdivs(path)
    else:
        score = load_others(path)
    return score


def main(args):
    models = ['bert', 'roberta', 'albert']
    datasets = ['ss', 'cp']
    methods = ['aul', 'cps', 'sss', 'kldivs', 'jsdivs']
    all_score_dict = dict()
    for data in datasets:
        data_score_dict = dict()
        for method in methods:
            method_score_list = []
            for model in models:
                score = scoreing(data, method, model, args.sample_rate)
                method_score_list.append(score)
            data_score_dict[method] = method_score_list
        all_score_dict[data] = data_score_dict

    for k, v in all_score_dict.items():
        print(k)
        for method, score_list in v.items():
            print_line = f'{method}: '
            for s in score_list:
                print_line += f' {s}'
            print_line += f' => {round(sum(score_list), 2)}'
            print(print_line)

if __name__ == "__main__":
    args = parse_args()
    main(args)
