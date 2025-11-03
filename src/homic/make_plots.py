import time
import statistics
from scipy import stats
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd
from collections import Counter
#####################################################################################################
#####################################################################################################
#####################################################################################################
################################# Plotting the statistics for reads

def alignment_scores(scores, path):
    plt.hist(scores, bins=30, color='skyblue', edgecolor='black')
     
    # Adding labels and title
    plt.xlabel('Normalized alignment score')
    plt.ylabel('Frequency')
    plt.title('Mean value: ') # ,  statistics.mean(scores)
     
    # Display the plot
    fname = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig(path + "/" + fname + '_hist_nas.png')

def data_stats(read_start, read_qual, read_qual_len, figs_path):
    lens = list(map(len, read_qual))

    fig = plt.figure(figsize=(15, 10))
    
    gs = gridspec.GridSpec(2,2)
    
    ax = plt.subplot(gs[0, 0])
    plt.hist(lens, color = "skyblue", ec="deepskyblue", bins = 20)
    plt.text(-0.05, 1.1, "A", fontsize=16, fontweight="bold", transform=ax.transAxes)
    plt.ylabel("Frequency")
    plt.xlabel("Read length [bp]")
    
    ax = plt.subplot(gs[0, 1])
    plt.hist(read_start, color = "mediumseagreen", ec="seagreen", bins = 100)
    plt.text(-0.05, 1.1, "B", fontsize=16, fontweight="bold", transform=ax.transAxes)
    plt.ylabel("Frequency")
    plt.xlabel("Normalized read start pos. [bp]")

    read_qual_sel = [num for num in read_qual if len(num) == read_qual_len] # selected reads
    read_qual_sel = np.array(read_qual_sel)

    ax = plt.subplot(gs[1, :])
    bplot = plt.boxplot(read_qual_sel, patch_artist=True)
    plt.text(-0.05, 1.1, "C", fontsize=16, fontweight="bold", transform=ax.transAxes)
    plt.ylabel("Quality score")
    plt.xlabel("Read posistion for reads length " + str(read_qual_len))
    plt.xticks(color='w')
    
    colors = sns.color_palette("cubehelix", read_qual_len)
    
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    
    fig.tight_layout()
    
    # Save the plot
    fname = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig(figs_path + "/" + fname + '_data_overview.png')

def metrics(history, figs_path):
    fig = plt.figure(figsize=(15, 10))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    metrics = ['loss', 'categorical_accuracy', 'AUC', 'f1_score']
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()

        tra_res = np.array(history.history[metric])
        val_res = np.array(history.history['val_'+metric])
        
        if metric == 'loss':
            plt.subplot(2,2,n+1)
            plt.plot(history.epoch, tra_res, color=colors[0], label='Train')
            plt.plot(history.epoch, val_res, color=colors[1], linestyle="--", label='Val')
            plt.xlabel('Epoch')
            plt.ylabel(name)
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.subplot(2,2,n+1)
            plt.plot(history.epoch, tra_res, color=colors[0], label='Train')
            plt.plot(history.epoch, val_res, color=colors[1], linestyle="--", label='Val')
            plt.xlabel('Epoch')
            plt.ylabel(name)
            plt.ylim([0.5,1])
        elif metric == 'f1_score':
            plt.subplot(2,2,n+1)
            plt.boxplot(tra_res, patch_artist=True) #, positions=np.array(len(tra_res))*2.0-0.35)
            #plt.boxplot(val_res, patch_artist=True, positions=np.array(len(val_res))*2.0+0.35)
            plt.xlabel('Class')
            plt.ylabel(name)
            plt.ylim([0,1])
        else:
            plt.subplot(2,2,n+1)
            plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
            plt.plot(history.epoch, history.history['val_'+metric],
                 color=colors[1], linestyle="--", label='Val')
            plt.xlabel('Epoch')
            plt.ylabel(name)
            plt.ylim([0,1])
        
        plt.legend()
          # Save the plot
    fname = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig(figs_path + "/" + fname + '_metrics.png')

def bench_barplot(pearson_d, taxa_orders, path, mtype):

    p_stats = {}
    sd_stats = {}
    for itp, tp in enumerate(['DL model', 'Kraken2']):
        pl=[]
        cl=[]
        ol=[]
        fl=[]
        gl=[]
        sl=[]
        l = [pl, cl, ol, fl, gl, sl]
    
        for k,v in pearson_d.items():
    
            for i, tx in enumerate(taxa_orders[:-1]):
                try:
                    l[i].append(v[tx][itp])
                except KeyError:
                    continue
        mean_l = []
        sd_l = []
        for i, il in enumerate(l):
            mean_l.append(round(sum(il) / len(il), 2))
            sd_l.append(statistics.stdev(il))
        p_stats[tp] = mean_l
        sd_stats[tp] = sd_l
    
    p_stats['taxa_order'] = taxa_orders[:-1]
    sd_stats['taxa_order'] = taxa_orders[:-1]
    
    p_df = pd.DataFrame(p_stats)
    p_df= pd.melt(p_df, id_vars=['taxa_order'], value_vars=['DL model', 'Kraken2'])
    sd_df = pd.DataFrame(sd_stats)
    sd_df = pd.melt(sd_df, id_vars=['taxa_order'], value_vars=['DL model', 'Kraken2'])

    P_stats = p_df
    P_stats['sdw'] = sd_df['value']
    #P_stats = pd.concat(P_stats)
    P_stats.reset_index(inplace=True)
    # Plot
    # for a 2x3 subplot
    axes = []
    for row_idx in range(0,1):
        for col_idx in range(0,1):
            ax = plt.subplot2grid((1, 1), (row_idx, col_idx))
            axes.append(ax)
    ax=axes[0]
    P_stats = P_stats.iloc[::-1] # reverse, to have species last

    #sde = P_stats['sdw'].tolist()
    #print(sde)
    #yerrv = []

    # Append all elements from the first list
    #for val in sde:
    #    yerrv.append([val,val])

    #breakpoint()
    palette = {'Kraken2': "#91C0E7", 'DL model': "#F1C77E"}
    sns.barplot(data=P_stats, x='taxa_order', y='value', hue='variable', ax=ax, palette=palette, gap=0.1) #


    x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
    y_coords = [p.get_height() for p in ax.patches]

    for container in ax.containers:
        ax.bar_label(container, label_type='center')
        
    ax.set_xlabel("")
    ax.errorbar(x=x_coords[:-2], y=y_coords[:-2], yerr=P_stats['sdw'], fmt="none", c="k")
    # Legend
    L=ax.legend()
    L.get_texts()[0].set_text('Kraken2')
    L.get_texts()[1].set_text('Kraken2 + DL model')
    sns.move_legend(ax, "lower left")
        
    if mtype=="pearson":

        ax.set_ylabel("Average Pearson correlation coefficient")
        fname = time.strftime("%Y%m%d-%H%M%S")
        plt.savefig(path + fname + '_pearson.png')
        plt.clf()
        
    elif mtype=="accuracy":

        ax.set_ylabel("Average accuracy")
        ax.set_xlabel("")
        fname = time.strftime("%Y%m%d-%H%M%S")
        plt.savefig(path + fname + '_accuracy.png')
        plt.clf()
        
    elif mtype=="precision":

        ax.set_ylabel("Average precision")
        ax.set_xlabel("")
        fname = time.strftime("%Y%m%d-%H%M%S")
        plt.savefig(path + fname + '_precision.png')
        plt.clf()
        
    elif mtype=="recall":

        ax.set_ylabel("Average recall")
        ax.set_xlabel("")
        fname = time.strftime("%Y%m%d-%H%M%S")
        plt.savefig(path + fname + '_recall.png')
        plt.clf()
        
        

def bray_curtis_bench(braycurtis_d, taxa_orders, path):
    # Plot BC
    fig = plt.figure(figsize=(40,10))
    
    # for a 2x2 subplot
    axes = []
    for row_idx in range(0,2):
        for col_idx in range(0,3):
            ax = plt.subplot2grid((2, 3), (row_idx, col_idx))
            axes.append(ax)
            
    bc_df = pd.DataFrame.from_dict(braycurtis_d, orient='index')
    bc_df = bc_df.iloc[::-1]
    for i, tx in enumerate(reversed(taxa_orders[:-1])):
        
        df2 = pd.DataFrame(bc_df[tx].to_list(), columns=['DL model','Kraken2']).reset_index()
        ax = axes[i]
        df3 = pd.melt(df2, id_vars=['index'], value_vars=['DL model','Kraken2'])
        palette = {'Kraken2': "#91C0E7", 'DL model': "#F1C77E"}
        sns.lineplot(data=df3, x='index', y='value', hue='variable', ax=ax, palette=palette)
        ax.set_title(tx, fontsize=25)
        ax.set_xlabel("Spatial spot", fontsize=25)
        ax.tick_params(axis='both', which='major', labelsize=25)
        
        if i == 0:
            ax.set_ylabel("Bray-Curtis distance", fontsize=25)
        else:
            ax.set_ylabel("")
        
        # Legend
        L=ax.legend(fontsize=25)
        L.get_texts()[0].set_text('Kraken2 + DL model')
        L.get_texts()[1].set_text('Kraken2')
    
    plt.tight_layout()
    fname = time.strftime("%Y%m%d-%H%M%S")
        
    plt.savefig(path + fname + '_bray-curtis.png')

def relative_abundance_double(data1, data2, name1, name2):
    # plot data in stack manner of bar type
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    data1.T.plot(
        kind='bar',
        stacked=True,
        ax=axes[0],
        colormap='tab20'
    )
    axes[0].set_title(name1)
    axes[0].set_ylabel('Relative abundance')
    axes[0].set_xticks([])
    for patch, label in zip(axes[0].patches, data1.index):
        if label == 'Other':
            patch.set_facecolor('dimgray')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
    data2.T.plot(
        kind='bar',
        stacked=True,
        ax=axes[1],
        colormap='tab20'
    )
    axes[1].set_title(name2)
    axes[1].set_ylabel('Relative abundance')
    axes[1].set_xticks([])
    for patch, label in zip(axes[1].patches, data2.index):
        if label == 'Other':
            patch.set_facecolor('dimgray')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    axes[0].text(-0.1, 1.05, "A", transform=axes[0].transAxes,
            fontsize=16, fontweight='bold', va='top', ha='right')
    axes[1].text(-0.1, 1.05, "B", transform=axes[1].transAxes,
            fontsize=16, fontweight='bold', va='top', ha='right')
    
    
    plt.tight_layout()
    plt.show()
    return fig



def relative_abundance_single(data, data_name, thr = 0.01, ntop = 10):

    nr = round(data.shape[0] * thr)
    data2 = data.iloc[:nr]
    counts = Counter(data2["genus"])
    
    specs = list(counts.keys())
    freqs = list(counts.values())
    
    inds = np.argsort(freqs)
    specs = [specs[i] for i in inds]
    freqs = [freqs[i] for i in inds]
    
    species = (data_name)
    
    width = 0.5
    c = specs[-ntop:]
    v = freqs[-ntop:] 
    weight_counts = dict(zip(c, v))
    
    fig, ax = plt.subplots()
    bottom = np.zeros(3)
    
    for boolean, weight_count in weight_counts.items():
        p = ax.bar(species, weight_count, width, label=boolean, bottom=bottom)
        bottom += weight_count
    
    ax.set_title("Genus frequency")
    ax.legend(loc="upper right")
    
    plt.show()


def relative_abundance_multi(listed_data, data_names, n=10, level="genus"):
    
    # compute counts for each df in a loop
    counts = [Counter(df[level]) for df in listed_data]
    
    # create DataFrame from all counts
    df = pd.DataFrame(counts).fillna(0).astype(int)
    
    # convert to percentages by row
    df_percent_row = df.div(df.sum(axis=1), axis=0) * 100
    
    # sort columns by their total sum
    df_sorted = df_percent_row[df_percent_row.sum().sort_values(ascending=False).index]
    
    # split
    df_first = df_sorted.iloc[:, :n]
    df_others = df_sorted.iloc[:, n:]
    
    # create "Other" column by summing the rest
    others_col = df_others.sum(axis=1).rename("Other")
    
    # Concatenate into final DataFrame
    df_final = pd.concat([df_first, others_col], axis=1)
    
    dftop = df_final
    dcttop = dftop.to_dict('series')
    
    weight_counts = dcttop
    width = 0.5
    
    fig, ax = plt.subplots()
    bottom = np.zeros(3)
    
    
    colors_pastel = [
        "#FF9999",  # Bright Pastel Red
        "#99FF99",  # Bright Pastel Green
        "#9999FF",  # Bright Pastel Blue
        "#FFCC99",  # Bright Pastel Orange
        "#CC99FF",  # Bright Pastel Purple
        "#99FFFF",  # Bright Pastel Cyan
        "#FFFF99",  # Bright Pastel Yellow
        "#FF99CC",  # Bright Pastel Pink
        "#FFCC66",  # Bright Pastel Peach
        "#99FFCC",  # Bright Pastel Mint
        "#CCCCCC",  # Bright Pastel Gray
    ]
    
    # function to darken a color
    def darken_color(hex_color, factor=0.65):
        rgb = mcolors.hex2color(hex_color)
        darkened = tuple([max(0, c * factor) for c in rgb])
        return mcolors.to_hex(darkened)
    
    # generate darker colors
    dark_pastels = [darken_color(c, 0.75) for c in colors_pastel]
    
    for (boolean, weight_count), color in zip(weight_counts.items(), dark_pastels):
        p = ax.bar(data_names, weight_count, width, label=boolean, bottom=bottom, color=color)
        bottom += weight_count
    
    
    ax.set_ylabel("Relative abundance (%)")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.show()

def three_corplots(listed_data, data_names, level="genus"):

    # compute counts for each df in a loop
    if isinstance(listed_data, list):
        counts = [Counter(df[level]) for df in listed_data]
            
        # create DataFrame from all counts
        df = pd.DataFrame(counts).fillna(0).astype(int)
        
        df = df.T
        df.columns = data_names
    else:
        df = listed_data

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1
    g = sns.regplot(x=data_names[0], y=data_names[1], data=df, ax=axes[0], scatter_kws={'s':20, 'color': 'black'}, line_kws={'color': 'black'})
    res = stats.pearsonr(df[data_names[0]], df[data_names[1]])
    
    axes[0].text(
        0.05, 0.95,
        f"r = {res[0]:.2f}\n"
        f"p = {res[1]:.3e}",
        transform=axes[0].transAxes,
        ha="left", va="top",
        fontsize=12, weight="bold"
        )
    
    # 2
    g = sns.regplot(x=data_names[0], y=data_names[2], data=df, ax=axes[1], scatter_kws={'s':20, 'color': 'black'}, line_kws={'color': 'black'})
    res = stats.pearsonr(df[data_names[0]], df[data_names[2]])
    
    axes[1].text(
        0.05, 0.95,
        f"r = {res[0]:.2f}\n"
        f"p = {res[1]:.3e}",
        transform=axes[1].transAxes,
        ha="left", va="top",
        fontsize=12, weight="bold"
        )

    # 3
    g = sns.regplot(x=data_names[1], y=data_names[2], data=df, ax=axes[2], scatter_kws={'s':20, 'color': 'black'}, line_kws={'color': 'black'})
    res = stats.pearsonr(df[data_names[1]], df[data_names[2]])
    
    axes[2].text(
        0.05, 0.95,
        f"r = {res[0]:.2f}\n"
        f"p = {res[1]:.3e}",
        transform=axes[2].transAxes,
        ha="left", va="top",
        fontsize=12, weight="bold"
        )

    plt.tight_layout()
    plt.show()

    
#print("Done!")
#print("Average length: ")
#print(stats.mean(lns))
#print(n_reads)
#print(lns_all)
#coverage_data = (stats.mean(lns)*n_reads)/sum(lns_all) # Lander-Waterman L*N/G
#print("Simulated data coverage: ")
#print(coverage_data)