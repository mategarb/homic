import time
import math
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
from .process_data import read_n_clean_blastn, chao_idx, goods_coverage

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


# Function to reorder dataframe so 'Other' is at the bottom
def reorder_other_first(df):
    species_order = ['Other'] + [s for s in df.index if s != 'Other']
    return df.reindex(species_order)


def relative_abundance_double(data1, data2, name1, name2):

    
    # Combine all species from both datasets
    all_species = list(set(data1.index) | set(data2.index))
    
    # Generate a color palette (excluding 'Other')
    palette = sns.color_palette('tab20', n_colors=len(all_species))
    color_dict = {species: color for species, color in zip(all_species, palette)}
    
    # Set 'Other' to dim gray
    color_dict['Other'] = 'dimgray'
    
    
    # Reorder datasets
    data1_reordered = reorder_other_first(data1)
    data2_reordered = reorder_other_first(data2)
    
    # Create figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot first dataset
    data1_reordered.T.plot(
        kind='bar',
        stacked=True,
        ax=axes[0],
        color=[color_dict[species] for species in data1_reordered.index]
    )
    axes[0].set_title(name1)
    axes[0].set_ylabel('Relative abundance')
    axes[0].set_xticks([])
    
    # Clean up spines
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['left'].set_visible(True)
    axes[0].spines['bottom'].set_visible(False)
    axes[0].tick_params(left=True, bottom=False, labelleft=True, labelbottom=False)
    
    # Reverse legend order to match stack
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot second dataset
    data2_reordered.T.plot(
        kind='bar',
        stacked=True,
        ax=axes[1],
        color=[color_dict[species] for species in data2_reordered.index]
    )
    axes[1].set_title(name2)
    axes[1].set_ylabel('Relative abundance')
    axes[1].set_xticks([])
    
    # Clean up spines
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['left'].set_visible(True)
    axes[1].spines['bottom'].set_visible(False)
    axes[1].tick_params(left=True, bottom=False, labelleft=True, labelbottom=False)
    
    # Reverse legend order
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add subplot labels
    axes[0].text(-0.1, 1.05, "A", transform=axes[0].transAxes,
                 fontsize=16, fontweight='bold', va='top', ha='right')
    axes[1].text(-0.1, 1.05, "B", transform=axes[1].transAxes,
                 fontsize=16, fontweight='bold', va='top', ha='right')
    
    plt.tight_layout()
    plt.show()

    return fig

def pastelize(color, factor=0.5):
    """Blend a color with white. factor=0 returns original color, factor=1 is full white"""
    rgb = mcolors.to_rgb(color)
    pastel_rgb = [(1 - factor) * c + factor * 1 for c in rgb]
    return pastel_rgb



def relative_abundance_four(data_list, bar_subtitles, common_legend=True, clegend_nrows=3):
    
    """
    Plot four stacked bar charts in a single row with:
    - 'Other' at bottom
    - legends (common or individual)
    - custom x-tick labels under each bar (rotated 45°)
    - subplots A and B labeled
    - y-axis goes to 1
    - bigger y-axis tick labels
    - polished spacing
    - common legend wrapped in 3 rows
    """
    # combine all species
    #all_species = list(set().union(*(df.index for df in data_list)))
    # Combine all species and sort alphabetically
    all_species = sorted(set().union(*(df.index for df in data_list)))

    palette = [
    '#FF0000', '#00FF00', '#0000FF', '#00FFFF', '#FF00FF', '#FFFF00',  # basic colors
    '#800000', '#008000', '#000080', '#008080', '#800080', '#808000',
    '#FFA500', '#A52A2A', '#5F9EA0', '#D2691E', '#DC143C', '#006400',
    '#8B008B', '#B8860B', '#556B2F', '#FF1493', '#1E90FF', '#FF4500',
    '#2E8B57', '#DAA520', '#00CED1', '#FF69B4', '#8A2BE2', '#7FFF00',
    '#FF6347', '#4682B4', '#9ACD32', '#FF8C00', '#6A5ACD', '#20B2AA',
    '#FFB6C1', '#8FBC8F', '#DDA0DD', '#00FA9A', '#FF7F50', '#6495ED'
    ]
    # Create pastel palette
    pastel_palette = [pastelize(c, factor=0.5) for c in palette]

    # Map species to colors (first 40 species)
    color_dict = {species: pastel_palette[i] for i, species in enumerate(all_species)}

    font_size = 20
    # Optional: make 'Other' always gray
    color_dict['Other'] = '#999999'
    
    
    data_list = [reorder_other_first(df) for df in data_list]

    fig, axes = plt.subplots(1, 4, figsize=(20, 18), sharey=True)
    
    for i, (df, subtitle, ax) in enumerate(zip(data_list, bar_subtitles, axes)):

        df.T.plot(kind='bar',
                  stacked=True,
                  ax=ax,
                  color=[color_dict[sp] for sp in df.index],
                  width=0.5,
                  legend=False)
        
        # Replace x-tick labels with given subtitle
        ax.set_xticks(range(df.shape[1]))
        ax.set_xticklabels([subtitle]*df.shape[1], rotation=45, ha='right', fontsize=font_size)
        
        # Force y-axis from 0 to 1
        ax.set_ylim(0, 1)
        
        # Increase y-axis tick label font size
        ax.tick_params(axis='y', labelsize=font_size)
        
        # Y-axis label only for first subplot
        if i == 0:
            ax.set_ylabel('Relative abundance', fontsize=font_size+2)
        else:
            ax.set_ylabel('')
        
        # Clean spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        
        # Individual legend if common_legend is False
        if not common_legend:
            handles = [plt.Rectangle((0,0),1,1,color=color_dict[sp]) for sp in df.index]
            ax.legend(handles[::-1], df.index[::-1], bbox_to_anchor=(1.05,1),
                      loc='upper left', fontsize=font_size)
    
    # add common legend if requested
    if common_legend:
        # sort alphabetically
        labels = sorted(all_species, reverse=True)
        handles = [plt.Rectangle((0,0),1,1,color=color_dict[sp]) for sp in labels]

        # Wrap legend in N rows
        ncol = len(all_species) // clegend_nrows + (len(all_species) % clegend_nrows > 0)

        fig.legend(handles[::-1], labels[::-1], loc='lower center',
                   ncol=ncol, fontsize=font_size, frameon=False, bbox_to_anchor=(0.5, -0.1), borderaxespad=0)
        plt.tight_layout(rect=[0,0.15,1,1])  # leave space at bottom

    else:
        plt.tight_layout()
    
    axes[0].text(-0.15, 1.05, "A", transform=axes[0].transAxes,
                 fontsize=font_size+4, fontweight='bold', va='top', ha='right')
    axes[2].text(-0.15, 1.05, "B", transform=axes[2].transAxes,
                 fontsize=font_size+4, fontweight='bold', va='top', ha='right')
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.2)
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

def saturation_curves(samp_list, reads_nos, path_in, path_out,
                      evalue=1e-200, pident=0.99, taxa="species"):

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Generate 30 distinct colors ---
    colors = (
        plt.cm.tab20(np.linspace(0, 1, 20)).tolist()
        + plt.cm.tab20b(np.linspace(0, 1, 10)).tolist()
    )

    for i, samp in enumerate(samp_list):
        reads_n = reads_nos[samp]
        species_num = []
        chao_all = []
        coverage_all = []
        
        for n in reads_n:
            path_blast = f"{path_in}{samp}/{samp}_{n}_blastn_report.txt"
            df = read_n_clean_blastn(path_blast, evalue=evalue, pident=pident)

            species_num.append(len(set(df[taxa])))
            chao_all.append(chao_idx(df[taxa])[0])

            cov, F1, N = goods_coverage(df[taxa])
            coverage_all.append(cov)

        color = colors[i % len(colors)]

        axes[0].plot(np.multiply(reads_n, 2), species_num,
                     color=color, lw=1.5, alpha=0.9, label=samp)

        axes[1].plot(np.multiply(reads_n, 2), chao_all, #also: coverage_all
                     color=color, lw=1.5, alpha=0.9)

    # --- Labels ---
    axes[0].set(xlabel="Reads number", ylabel="Unique species number")
    axes[1].set(xlabel="Reads number", ylabel="Chao1 index")

    # --- Panel labels ---
    axes[0].text(-0.08, 1.05, "A", transform=axes[0].transAxes,
                 fontsize=14, fontweight="bold")
    axes[1].text(-0.08, 1.05, "B", transform=axes[1].transAxes,
                 fontsize=14, fontweight="bold")

    # --- Single compact legend at bottom ---
    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=6,                # 30 samples → 5 rows of 6
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.12)
    )

    plt.tight_layout(rect=[0, 0.1, 1, 1])

    plt.savefig(path_out, dpi=300, bbox_inches="tight")
    plt.show()


def multi_saturation_curves(
    samp_list, reads_nos, path_in, path_out,
    evalue=1e-50, pident=0.99, taxa="species"
):

    n_samples = len(samp_list)
    n_cols = 5                       # 5*6 = 30 max
    n_rows = math.ceil(n_samples / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(15, 3 * n_rows),
        sharex=True, sharey=True
    )
    axes = axes.flatten()

    dark_blue = "#1f4fa3"

    for i, samp in enumerate(samp_list):
        ax = axes[i]
        reads_n = reads_nos[samp]

        species_num = []

        for n in reads_n:
            path_blast = f"{path_in}{samp}/{samp}_{n}_blastn_report.txt"
            df = read_n_clean_blastn(path_blast, evalue=evalue, pident=pident)
            species_num.append(len(set(df[taxa])))

        x = np.multiply(reads_n, 2)

        ax.plot(x, species_num, color=dark_blue, lw=2)

        # Sample name
        ax.text(
            0.03, 0.95, samp,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=9, fontweight="bold"
        )

        ax.tick_params(labelsize=8)

    # Hide unused panels
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    # Shared labels
    fig.text(0.5, 0.04, "Reads", ha="center", fontsize=14)
    fig.text(0.04, 0.5, "Unique species", va="center", rotation="vertical", fontsize=14)

    plt.tight_layout(rect=[0.06, 0.06, 1, 1])

    # Save and show
    fig.savefig(path_out, dpi=300, bbox_inches="tight")
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