#plt.hist(all_scores, bins=30, color='skyblue', edgecolor='black')
 
# Adding labels and title
#plt.xlabel('Normalized alignment score')
#plt.ylabel('Frequency')
#plt.title('Basic Histogram')
 
# Display the plot
#plt.show()
#plt.savefig('hist_nas.png')


#####################################################################################################
#####################################################################################################
#####################################################################################################
################################# Plotting the statistics for reads

#plt.hist(all_scores, bins=30, color='skyblue', edgecolor='black')
 
# Adding labels and title
#plt.xlabel('Normalized alignment score')
#plt.ylabel('Frequency')
#plt.title('Basic Histogram')
 
# Display the plot
#plt.show()
#plt.savefig('hist_nas.png')

#lns = [len(sub) if type(sub) != int else 1 for sub in r2_read_lines]
#lns_all = [len(sub) if type(sub) != int else 1 for sub in sp_read_length_dict.values()]
#qualities_150_ar = np.array(qualities_150)

#fig = plt.figure(figsize=(15, 10))

#gs = gridspec.GridSpec(2,2)

#ax = plt.subplot(gs[0, 0])
#plt.hist(lns, color = "skyblue", ec="deepskyblue", bins = 20)
#plt.text(-0.05, 1.1, "A", fontsize=16, fontweight="bold", transform=ax.transAxes)
#plt.ylabel("Frequency")
#plt.xlabel("Read length [bp]")

#ax = plt.subplot(gs[0, 1])
#plt.hist(start_vec, color = "mediumseagreen", ec="seagreen", bins = 100)
#plt.text(-0.05, 1.1, "B", fontsize=16, fontweight="bold", transform=ax.transAxes)
#plt.ylabel("Frequency")
#plt.xlabel("Normalized read start pos. [bp]")

#ax = plt.subplot(gs[1, :])
#bplot = plt.boxplot(qualities_150_ar, patch_artist=True)
#plt.text(-0.05, 1.1, "C", fontsize=16, fontweight="bold", transform=ax.transAxes)
#plt.ylabel("Quality score")
#plt.xlabel("Read pos. for 150bp")
#plt.xticks(color='w')

#colors = sns.color_palette("cubehelix", 151)

#for patch, color in zip(bplot['boxes'], colors):
#    patch.set_facecolor(color)

    
#fig.tight_layout()

# Display the plot
#plt.savefig('figs/simulated_D2_overview.png')
        
#print("Done!")
#print("Average length: ")
#print(stats.mean(lns))
#print(n_reads)
#print(lns_all)
#coverage_data = (stats.mean(lns)*n_reads)/sum(lns_all) # Lander-Waterman L*N/G
#print("Simulated data coverage: ")
#print(coverage_data)