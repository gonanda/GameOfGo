import os
import h5py
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
    with open(work_dir+'stats-nanda.json') as infile:
        statsnanda = json.load(infile)
    return concat_duplicates(stats+statsnanda)

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

def plot_refs(refs,agents,quantity):
    fig, ax = plt.subplots()
    fig.set_facecolor('w')
    agent_list = [ag['agent'] for ag in agents] 
    ref_list = [ref[quantity] for ref in refs]
    if len(agent_list)<len(ref_list):
        ref_list.pop()
    ax.plot(agent_list,ref_list)

def load_agents(work_dir, directory):
    with open(work_dir+directory+'/agents.json') as infile:
        agents = json.load(infile)
    return agents

def plot_loss(agents,quantity):
    fig, ax = plt.subplots()
    fig.set_facecolor('w')
    x_list =  [agents[0]['agent']/agents[0]['ep']*j for j in range(1,agents[0]['ep']+1)]
    y_list = agents[0]['history'][quantity].copy()
    for i in range(1,len(agents)):
        x_list += [agents[i-1]['agent']+(agents[i]['agent']-agents[i-1]['agent'])/agents[i]['ep']*j for j in range(1,agents[i]['ep']+1)]
        y_list += agents[i]['history'][quantity]
    ax.scatter(x_list, y_list)

def plot_all_loss(ag):
    fig, (ax1,ax2,ax3) = plt.subplots(3)
    fig.set_facecolor('w')
    x_list =  [ag[0]['agent']/ag[0]['ep']*j for j in range(1,ag[0]['ep']+1)]
    y_list_total = ag[0]['history']['loss'].copy()
    y_list_policy = ag[0]['history']['policy_head_dense_loss'].copy()
    y_list_value = ag[0]['history']['value_head_output_loss'].copy()
    for i in range(1,len(ag)):
        x_list += [ag[i-1]['agent']+(ag[i]['agent']-ag[i-1]['agent'])/ag[i]['ep']*j for j in range(1,ag[i]['ep']+1)]
        y_list_total += ag[i]['history']['loss']
        y_list_policy += ag[i]['history']['policy_head_dense_loss']
        y_list_value += ag[i]['history']['value_head_output_loss']
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax3.set_yscale('log')
    ax1.scatter(x_list, y_list_total)
    ax2.scatter(x_list, y_list_policy)
    ax3.scatter(x_list, y_list_value)


def filter_two_nums(stats,ag_dir,ag_b_num,ag_w_num,nr,c,cp,dw):
    temp_stats = stats
    temp_stats = [res for res in temp_stats if res['ag_b_dir']==ag_dir and res['ag_w_dir']==ag_dir]
    temp_stats = [res for res in temp_stats if res['ag_b_num']==ag_b_num and res['ag_w_num']==ag_w_num]
    temp_stats = [res for res in temp_stats if res['nr_b']==nr and res['nr_w']==nr]
    temp_stats = [res for res in temp_stats if res['c_b']==c and res['c_w']==c]
    temp_stats = [res for res in temp_stats if res['cp_b']==cp and res['cp_w']==cp]
    temp_stats = [res for res in temp_stats if res['dw_b']==dw and res['dw_w']==dw]
    return temp_stats[0]['scores']
    
def plot_training_scores(stats,directory,refs,agents,nr=None,c=None,cp=None,dw=None,alpha=0.05,disp=10.):
    agents_list=[0]
    agents_plot_list=[]
    for ag in agents:
        agents_list += [ag['agent']]
        agents_plot_list += [ag['agent']]
    references_list=[0]
    for ref in refs:
        references_list += [ref['agent']]
        
    bl_ref_list = []
    wh_ref_list = []
    
    for i in range(len(agents_plot_list)):
        bl_ref_list += [(agents_plot_list[i],filter_two_nums(stats,ag_dir=directory,ag_b_num=references_list[i],ag_w_num=agents_list[i],nr=nr,c=c,cp=cp,dw=dw))]
        wh_ref_list += [(agents_plot_list[i],[-x for x in filter_two_nums(stats,ag_dir=directory,ag_b_num=agents_list[i],ag_w_num=references_list[i],nr=nr,c=c,cp=cp,dw=dw)])]
        
    fig, ax = plt.subplots()
    fig.set_facecolor('w')
    for tup in bl_ref_list:
        ax.scatter([tup[0]-disp]*len(tup[1]),tup[1],alpha=alpha,color='b')
    
    for tup in wh_ref_list:
        ax.scatter([tup[0]+disp]*len(tup[1]),tup[1],alpha=alpha,color='r')

def plot_benchmark(work_dir_black,work_dir_white):
    split_work_dir_black = os.path.split(os.path.abspath(work_dir_black))
    split_work_dir_white = os.path.split(os.path.abspath(work_dir_white))
    results_path = os.path.join(split_work_dir_black[0],'%s-%s-results.json' % (split_work_dir_black[1],split_work_dir_white[1]))
    ag_path_black = os.path.join(work_dir_black,'agents.json')
    ag_path_white = os.path.join(work_dir_white,'agents.json')

    with open(results_path) as infile:
        benchmark = json.load(infile)

    with open(ag_path_black) as infile:
        ag_data_black = json.load(infile)
    agent_nums_black = [0] + [entry['agent'] for entry in ag_data_black]

    with open(ag_path_white) as infile:
        ag_data_white = json.load(infile)
    agent_nums_white = [0] + [entry['agent'] for entry in ag_data_white]

    with h5py.File(os.path.join(work_dir_black,'agent_00000000.hdf5'), 'r') as h5file:
        board_size = int(h5file['encoder'].attrs['board_size'])

    def res_col(res):
        if res is None or res == 'wip':
            return (1.,1.,1.)
        elif res>0:
            return (1.*res/board_size**2,0,0)
        elif res<0:
            return (0,-1.*res/board_size**2,0)
        else:
            return (0,0,1.)

    result_array = [[res_col(benchmark.get('%s-%s' % (b_num,w_num))) for w_num in agent_nums_white] for b_num in agent_nums_black]

    fig, ax = plt.subplots()
    fig.set_facecolor('w')

    ax.imshow(result_array, vmin=0, vmax=board_size**2)

    
def plot_benchmark_sym(work_dir_1,work_dir_2,clip_max,clip_min):
    split_work_dir_1 = os.path.split(os.path.abspath(work_dir_1))
    split_work_dir_2 = os.path.split(os.path.abspath(work_dir_2))
    results_path_A = os.path.join(split_work_dir_1[0],'%s-%s-results.json' % (split_work_dir_1[1],split_work_dir_2[1]))
    results_path_B = os.path.join(split_work_dir_1[0],'%s-%s-results.json' % (split_work_dir_2[1],split_work_dir_1[1]))
    ag_path_1 = os.path.join(work_dir_1,'agents.json')
    ag_path_2 = os.path.join(work_dir_2,'agents.json')

    with open(results_path_A) as infile:
        benchmark_A = json.load(infile)
        
    with open(results_path_B) as infile:
        benchmark_B = json.load(infile)

    with open(ag_path_1) as infile:
        ag_data_1 = json.load(infile)
    agent_nums_1 = [0] + [entry['agent'] for entry in ag_data_1]

    with open(ag_path_2) as infile:
        ag_data_2 = json.load(infile)
    agent_nums_2 = [0] + [entry['agent'] for entry in ag_data_2]

    with h5py.File(os.path.join(work_dir_1,'agent_00000000.hdf5'), 'r') as h5file:
        board_size = int(h5file['encoder'].attrs['board_size'])

    def res_col(resA,resB):
        if resA is None or resA == 'wip' or resB is None or resB == 'wip':
            return (1.,1.,1.)
        elif resA-resB>clip_max:
            return (1.,0,0)
        elif clip_max>resA-resB>clip_min:
            return (1.*(resA-resB)/(clip_max-clip_min),0,0)
        elif resB-resA>clip_max:
            return (0,1.,0)
        elif clip_max>resB-resA>clip_min:
            return (0,1.*(resB-resA)/(clip_max-clip_min),0)
        else:
            return (0,0,1.)

    result_array = [[res_col(benchmark_A.get('%s-%s' % (num_1,num_2)),benchmark_B.get('%s-%s' % (num_2,num_1))) for num_2 in agent_nums_2] for num_1 in agent_nums_1]

    fig, ax = plt.subplots()
    fig.set_facecolor('w')

    ax.imshow(result_array, vmin=0, vmax=board_size**2)
    fig.savefig("plot_%s_%s.svg" % (split_work_dir_1[1],split_work_dir_2[1]))