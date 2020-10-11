import json
import numpy as np
import statistics as sta
import matplotlib.pyplot as plt

in_fields = ['ag_b_dir','ag_b_num','ag_w_dir','ag_w_num','nr_b','nr_w','c_b','c_w','cp_b','cp_w','dw_b','dw_w']

def score_lengths(stats_list):
    sc_len = []
    for dic in stats_list:
        sc_len += [len(dic['scores'])]
    return sc_len        

def compare_dicts(dict1, dict2):
    equal = True
    for field in in_fields:
        if dict1[field] != dict2[field]:
            equal = False
            break
    return equal
        
def concat_duplicates(stats_list):
    new_list = []
    while len(stats_list)>0:
        indices=[0]
        if len(stats_list)>1:
            for i in range(1,len(stats_list)):
                if compare_dicts(stats_list[0],stats_list[i]):
                    indices.append(i)
        for i in range(1,len(indices)):
            stats_list[0]['scores'] += stats_list[i]['scores']
        new_list += [stats_list[0]]
        for i in range(len(indices)-1,-1,-1):
            stats_list.pop(i)
    return new_list

def load_stats(work_dir):
    with open(work_dir+'stats.json') as infile:
        stats = json.load(infile)
    return concat_duplicates(stats)

def filter_sym(stats,ag_dir=None,ag_num=None,nr=None,c=None,cp=None,dw=None):
    temp_stats = stats
    if ag_dir is not None:
        temp_stats = [res for res in temp_stats if res['ag_b_dir']==ag_dir and res['ag_w_dir']==ag_dir]
    else:
        temp_stats = [res for res in temp_stats if res['ag_b_dir']==res['ag_w_dir']]
    if ag_num is not None:
        temp_stats = [res for res in temp_stats if res['ag_b_num']==ag_num and res['ag_w_num']==ag_num]
    else:
        temp_stats = [res for res in temp_stats if res['ag_b_num']==res['ag_w_num']]
    if nr is not None:
        temp_stats = [res for res in temp_stats if res['nr_b']==nr and res['nr_w']==nr]
    else:
        temp_stats = [res for res in temp_stats if res['nr_b']==res['nr_w']]
    if c is not None:
        temp_stats = [res for res in temp_stats if res['c_b']==c and res['c_w']==c]
    else:
        temp_stats = [res for res in temp_stats if res['c_b']==res['c_w']]
    if cp is not None:
        temp_stats = [res for res in temp_stats if res['cp_b']==cp and res['cp_w']==cp]
    else:
        temp_stats = [res for res in temp_stats if res['cp_b']==res['cp_w']]
    if dw is not None:
        temp_stats = [res for res in temp_stats if res['dw_b']==dw and res['dw_w']==dw]
    else:
        temp_stats = [res for res in temp_stats if res['dw_b']==res['dw_w']]
    return temp_stats

def bin_p_and_err(x, n, suc, z):
    return (x, suc/n, suc/n-(suc+0.5*z*z)/(n+z*z)+z/(n+z*z)*np.sqrt(suc*(n-suc)/n+z*z/4), (suc+0.5*z*z)/(n+z*z)+z/(n+z*z)*np.sqrt(suc*(n-suc)/n+z*z/4)-suc/n)

def plot_sym(stats,ag_dir=None,ag_num=None,nr=None,c=None,cp=None,dw=None,x=None,alpha=0.05):
        fil_sta=filter_sym(stats,ag_dir=ag_dir,ag_num=ag_num,nr=nr,c=c,cp=cp,dw=dw)
        fig, ax = plt.subplots()
        fig.set_facecolor('w')
        tup_list = [(sta[x],sta['scores']) for sta in fil_sta]
        for tup in tup_list:
            ax.scatter([tup[0]]*len(tup[1]),tup[1],alpha=alpha,color='b')
            
def load_refs(work_dir, directory):
    with open(work_dir+directory+'/references.json') as infile:
        refs = json.load(infile)
    return refs

def plot_refs(refs,quantity):
    fig, ax = plt.subplots()
    fig.set_facecolor('w')
    ax.plot([ref['agent'] for ref in refs],[ref[quantity] for ref in refs])
