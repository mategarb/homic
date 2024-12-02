import time
import statistics
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
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
        
#print("Done!")
#print("Average length: ")
#print(stats.mean(lns))
#print(n_reads)
#print(lns_all)
#coverage_data = (stats.mean(lns)*n_reads)/sum(lns_all) # Lander-Waterman L*N/G
#print("Simulated data coverage: ")
#print(coverage_data)