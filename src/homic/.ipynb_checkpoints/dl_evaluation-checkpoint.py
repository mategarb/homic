import pickle
import os
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
from numpy import argmax
from scipy.spatial import distance
import statistics
from functools import reduce
from scipy import stats
import warnings
from matplotlib.text import Text
import matplotlib.pyplot as plt
from matplotlib import transforms, lines
import matplotlib.transforms as mtransforms
from matplotlib.font_manager import FontProperties
from seaborn.utils import remove_na
import statsmodels.stats.multitest as multi
from scipy.stats import shapiro 
from scipy import stats
from tqdm import tqdm
from tensorflow.keras import Model
from sklearn import metrics

warnings.filterwarnings('ignore')
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42



def dna_encode_embedding_table(dna_input, name="dna_encode"):
    embedding_values = np.zeros([len(dna_input), 5], np.float32)
    values = ("A", "C", "G", "T", "N")
    for j, b in enumerate(dna_input):
        if b in values:
            embedding_values[j, values.index(b)] = 1
    return embedding_values


#def extract_taxa_info(info_xy, column_header): that is just two liner, has been embedded with main code 
#    y_taxaorder = info_xy[column_header].tolist()
#    y_fastqH = info_xy['fastq'].tolist()
    
#    return y_taxaorder, y_fastqH

#def predict_taxa_in_model(Xpad, model): that is just one line, has been embedded with main code 
#    predictions = model.predict(Xpad)
    
#    return predictions

def calculate_missclassified(sant, forutspadd):
    """
    Calculates True postives, false negatives, # NAs and misclassifed taxa when comparing predicted taxa with the 
    groud truth. 
    """
    # percentage of misclassified examples
    if len(sant) != len(forutspadd):
        print('Ground truth and predicted are not the same length.')
        
    true_positives = 0
    missclassified_count = 0
    NA_count = 0
    false_positive = 0
    for pp, prediction in enumerate(forutspadd):
        if prediction == sant[pp]:
            true_positives += 1
        else:
            missclassified_count +=1
            if prediction == 'NA':
                NA_count +=1
            else:
                false_positive +=1

    # Percetage
    TP = true_positives / len(forutspadd) 
    FN = false_positive / len(forutspadd)
    NA = NA_count / len(forutspadd)
    misclassified = missclassified_count / len(forutspadd)
    
    return TP, FN, NA, misclassified

def change_order(row):
        rv_row = row.split(',')
        rv_row.reverse()
        return ','.join(rv_row) # return species level

def rm_specie(row):
    rv_row = row.split(',')
    if len(rv_row) == 7:
        return ','.join(rv_row[1:]) # return genus level
    else:
        return ','.join(rv_row) # return genus level

DEFAULT = object()

def stat_test(box_data1, box_data2, test):
    test_short_name = ''
    formatted_output = None
    if test == 'Wilcoxon':
        u_stat, pval = stats.ranksums(box_data1, box_data2)
        test_short_name = 'Wilcox'
        formatted_output = ("Wilcoxon test P_val={:.3e} U_stat={:.3e}"
                           .format(pval, u_stat))
        
    elif test == 'ttest':
        u_stat, pval = stats.ttest_ind(box_data1, box_data2, equal_var=True)
        test_short_name = 'T-test'
        formatted_output = ("T-test P_val={:.3e} U_stat={:.3e}"
                           .format(pval, u_stat))
    return pval, u_stat,formatted_output, test_short_name


def pval_annotation_text(x, pvalue_thresholds):
    singleValue = False
    if type(x) is np.array:
        x1 = x
    else:
        x1 = np.array([x])
        singleValue = True
    # Sort the threshold array
    pvalue_thresholds = pd.DataFrame(pvalue_thresholds).sort_values(by=0, ascending=False).values
    x_annot = pd.Series(["" for _ in range(len(x1))])
    for i in range(0, len(pvalue_thresholds)):
        if (i < len(pvalue_thresholds)-1):
            condition = (x1 <= pvalue_thresholds[i][0]) & (pvalue_thresholds[i+1][0] < x1)
            x_annot[condition] = pvalue_thresholds[i][1]
        else:
            condition = x1 < pvalue_thresholds[i][0]
            x_annot[condition] = pvalue_thresholds[i][1]

    return x_annot if not singleValue else x_annot.iloc[0]


def simple_text(pval, pvalue_format, pvalue_thresholds, test_short_name=None):
    """
    Generates simple text for test name and pvalue
    :param pval: pvalue
    :param pvalue_format: format string for pvalue
    :param test_short_name: Short name of test to show
    :param pvalue_thresholds: String to display per pvalue range
    :return: simple annotation
    """
    # Sort thresholds
    thresholds = sorted(pvalue_thresholds, key=lambda x: x[0])

    # Test name if passed
    text = test_short_name and test_short_name + " " or ""

    for threshold in thresholds:
        if pval < threshold[0]:
            pval_text = "p ≤ {}".format(threshold[1])
            break
    else:
        pval_text = "p = {}".format(pvalue_format).format(pval)

    #return text + pval_text
    return pval_text


def add_stat_annotation(ax,
                        data=None, x=None, y=None, hue=None, order=None, hue_order=None,
                        box_pairs=None,
                        test='t-test_welch', text_format='star', pvalue_format_string=DEFAULT,
                        loc='inside', show_test_name=True, pvalue_thresholds=DEFAULT,
                        use_fixed_offset=False, line_offset_to_box=None, line_offset=None,
                        line_height=0.02, text_offset=1, stack=True,
                        color='0.2', linewidth=1.5, fontsize='medium', verbose=1, multi_corr = False):
    """
    User should use the same argument for the data, x, y, hue, order, hue_order as the seaborn boxplot function.
    line_height is in axes fraction coordinates.
    text_offset is in points.
    box_pairs can be of either form:
    For non-grouped boxplot: [(cat1, cat2), (cat3, cat4)]
    For boxplot grouped by hue: [((cat1, hue1), (cat2, hue2)), ((cat3, hue3), (cat4, hue4))]
    Default pvalue_format_string is "{.3e}".
    pvalue_thresholds is a list of lists or tuples. Default is:
    For "star" text_format:
        [[1e-4, "****"], [1e-3, "***"], [1e-2, "**"], [0.05, "*"], [1, "ns"]]
    For "simple" text_format :
        [[1e-5, "1e-5"], [1e-4, "1e-4"], [1e-3, "0.001"], [1e-2, "0.01"]]
    """

    def find_x_position_box(box_plotter, boxName):
        """
        boxName can be either a name "cat" or a tuple ("cat", "hue")
        """
        if box_plotter.plot_hues is None:
            cat = boxName
            hueOffset = 0
        else:
            cat = boxName[0]
            hue = boxName[1]
            hueOffset = box_plotter.hue_offsets[box_plotter.hue_names.index(hue)]

        groupPos = box_plotter.group_names.index(cat)
        boxPos = groupPos + hueOffset
        return boxPos


    def get_box_data(box_plotter, boxName):
        """
        boxName can be either a name "cat" or a tuple ("cat", "hue")
        Here we really have to duplicate seaborn code, because there is not direct access to the
        box_data in the BoxPlotter class.
        """
        if box_plotter.plot_hues is None:
            cat = boxName
        else:
            cat = boxName[0]
            hue = boxName[1]

        i = box_plotter.group_names.index(cat)
        group_data = box_plotter.plot_data[i]

        if box_plotter.plot_hues is None:
            # Draw a single box or a set of boxes
            # with a single level of grouping
            box_data = remove_na(group_data)
        else:
            hue_level = hue
            hue_mask = box_plotter.plot_hues[i] == hue_level
            box_data = remove_na(group_data[hue_mask])

        return box_data

    # Set default values if necessary
    if pvalue_format_string is DEFAULT:
        pvalue_format_string = '{:.3e}'
        simple_format_string = '{:.2f}'
    else:
        simple_format_string = pvalue_format_string

    if pvalue_thresholds is DEFAULT:
        if text_format == "star":
            pvalue_thresholds = [[1e-4, "****"], [1e-3, "***"],
                                 [1e-2, "**"], [0.05, "*"], [1, "ns"]]
        else:
            pvalue_thresholds = [[1e-5, "1e-5"], [1e-4, "1e-4"],
                                 [1e-3, "0.001"], [1e-2, "0.01"]]
    
    fig = plt.gcf()

    valid_list = ['inside', 'outside']
    if loc not in valid_list:
        raise ValueError("loc value should be one of the following: {}.".format(', '.join(valid_list)))
    valid_list = ['Wilcoxon', 'ttest']
    if test not in valid_list:
        raise ValueError("test value should be one of the following: {}.".format(', '.join(valid_list)))

    if verbose >= 1 and text_format == 'star':
        print("pvalue annotation legend:")
        pvalue_thresholds = pd.DataFrame(pvalue_thresholds).sort_values(by=0, ascending=False).values
        for i in range(0, len(pvalue_thresholds)):
            if (i < len(pvalue_thresholds)-1):
                print('{}: {:.2e} < p <= {:.2e}'.format(pvalue_thresholds[i][1], pvalue_thresholds[i+1][0], pvalue_thresholds[i][0]))
            else:
                print('{}: p <= {:.2e}'.format(pvalue_thresholds[i][1], pvalue_thresholds[i][0]))
        print()

    # Create the same BoxPlotter object as seaborn's boxplot
    box_plotter = sns.categorical._BoxPlotter(x, y, hue, data, order, hue_order,
                                              orient=None, width=.8, color=None, palette=None, saturation=.75,
                                              dodge=True, fliersize=5, linewidth=None)
    ylim = ax.get_ylim()
    yrange = ylim[1] - ylim[0]

    if line_offset is None:
        if loc == 'inside':
            line_offset = 0.05
            if line_offset_to_box is None:
                line_offset_to_box = 0.06
        elif loc == 'outside':
            line_offset = 0.03
            line_offset_to_box = line_offset
    else:
        if loc == 'inside':
            if line_offset_to_box is None:
                line_offset_to_box = 0.06
        elif loc == 'outside':
            line_offset_to_box = line_offset
    y_offset = line_offset*yrange
    y_offset_to_box = line_offset_to_box*yrange

    y_stack = []
    annList = []
    test_result_list = []
    multi_corr_list = []
    for box1, box2 in box_pairs:

        valid = None
        group_names = box_plotter.group_names
        hue_names = box_plotter.hue_names
        if box_plotter.plot_hues is None:
            cat1 = box1
            cat2 = box2
            hue1 = None
            hue2 = None
            label1 = '{}'.format(cat1)
            label2 = '{}'.format(cat2)
            valid = cat1 in group_names and cat2 in group_names
        else:
            cat1 = box1[0]
            hue1 = box1[1]
            cat2 = box2[0]
            hue2 = box2[1]
            label1 = '{}_{}'.format(cat1, hue1)
            label2 = '{}_{}'.format(cat2, hue2)
            valid = cat1 in group_names and cat2 in group_names and hue1 in hue_names and hue2 in hue_names

        if valid:
            # Get position of boxes
            x1 = find_x_position_box(box_plotter, box1)
            x2 = find_x_position_box(box_plotter, box2)
            box_data1 = get_box_data(box_plotter, box1)
            box_data2 = get_box_data(box_plotter, box2)
            ymax1 = box_data1.max()
            ymax2 = box_data2.max()

            pval, u_stat,formatted_output, test_short_name = stat_test(box_data1, box_data2, test)
            test_result_list.append({'pvalue':pval, 'test_short_name':test_short_name,
                                   'formatted_output':formatted_output, 'box1':box1, 'box2':box2})
                
            if verbose >= 1:
                print("{} v.s. {}: {}".format(label1, label2, formatted_output))

            if text_format == 'full':
                text = "{} p = {}".format('{}', pvalue_format_string).format(test_short_name, pval)
            elif text_format is None:
                text = None
            elif text_format == 'star':
                text = pval_annotation_text(pval, pvalue_thresholds)
            elif text_format == 'simple':
                test_short_name = show_test_name and test_short_name or ""
                text = simple_text(pval, simple_format_string, pvalue_thresholds,
                                   test_short_name)

            if loc == 'inside':
                yref = max(ymax1, ymax2)
            elif loc == 'outside':
                yref = ylim[1]

            if stack:
                if len(y_stack) > 0:
                    yref2 = max(yref, max(y_stack))
                else:
                    yref2 = yref
            else:
                yref2 = yref

            if len(y_stack) == 0:
                y = yref2 + y_offset_to_box
            else:
                y = yref2 + y_offset
            h = line_height*yrange
            lineX, lineY = [x1, x1, x2, x2], [y, y + h, y + h, y]
            if loc == 'inside':
                ax.plot(lineX, lineY, lw=linewidth, c=color)
            elif loc == 'outside':
                line = lines.Line2D(lineX, lineY, lw=linewidth, c=color, transform=ax.transData)
                line.set_clip_on(False)
                ax.add_line(line)

            if text is not None:
                ann = ax.annotate(text, xy=(np.mean([x1, x2]), y + h),
                                  xytext=(0, text_offset), textcoords='offset points',
                                  xycoords='data', ha='center', va='bottom', fontsize=fontsize,
                                  clip_on=False, annotation_clip=False)
                annList.append(ann)

            ax.set_ylim((ylim[0], 1.1*(y + h)))

            if text is not None:
                plt.draw()
                y_top_annot = None
                got_mpl_error = False
                if not use_fixed_offset:
                    try:
                        bbox = ann.get_window_extent()
                        bbox_data = bbox.transformed(ax.transData.inverted())
                        y_top_annot = bbox_data.ymax
                    except RuntimeError:
                        got_mpl_error = True

                if use_fixed_offset or got_mpl_error:
                    if verbose >= 1:
                        print("Warning: cannot get the text bounding box. Falling back to a fixed y offset. Layout may be not optimal.")
                    # We will apply a fixed offset in points, based on the font size of the annotation.
                    fontsize_points = FontProperties(size='medium').get_size_in_points()
                    offsetTrans = mtransforms.offset_copy(ax.transData, fig=fig,
                                                          x=0, y=1.0*fontsize_points + text_offset, units='points')
                    yTopDisplay = offsetTrans.transform((0, y + h))
                    y_top_annot = ax.transData.inverted().transform(yTopDisplay)[1]
            else:
                y_top_annot = y + h

            y_stack.append(y_top_annot)
        else:
            raise ValueError("box_pairs contains an unvalid box pair.")
            pass

    y_stack_max = max(y_stack)
    if loc == 'inside':
        ax.set_ylim((ylim[0], 1.03*y_stack_max))
    elif loc == 'outside':
        ax.set_ylim((ylim[0], ylim[1]))
    
    if multi_corr == True:
        corrected_pvals = multiple_correction(multi_corr_list)
        
        for i, corr_p in enumerate(corrected_pvals):
            test_result_list[i]['Corrected pvalue'] = corr_p

    return ax, test_result_list


def taxa_per_spots(report_file, ft_file, fastq_file): # kraken 2 file, fastq header - taxa text file, fastq file
    #########################################################################
    #########################################################################
    # Load kraken2 report file
    
    #report_file = sys.argv[1]
    
    report = pd.read_csv(report_file, sep='\t', header=None, names=['fragments covered by clade (%)', 'fragments covered by clade (num)', 'fragments assigned (num)', 'torder', 'ncbi taxID', 'sci-name'])
    
    # Create a dictionary with the taxa order with taxID as key and taxa order as value + sp_dict = only taxID as value and specie name as key (for species only) 
    tax_order_dict, sp_dict = taxa_assignment(report)
    
    #########################################################################
    #########################################################################
    # Load fastq files
    
    # read input fasta file
    fastq = parser(fastq_file)
    
    #########################################################################
    #########################################################################
    # Load fastq header - taxa text file
    #ft_file = sys.argv[2]
    path = os.path.dirname(os.path.abspath(ft_file))
    base = os.path.basename(os.path.abspath(ft_file)).split('_headers')[0]
    
    ft = pd.read_csv(ft_file, sep='|', header = None, usecols=[0, 3, 4, 5, 6], names = ['fastq', 'x', 'y', 'UMI', 'taxa'])
    
    ft.set_index('fastq', inplace=True)
    
    # Remove fastq headers with short UMIs
    ft = ft[ft['UMI'].str.len().gt(11)]
    
    # Place whole taxa order per taxID
    ft['taxa_orderTMP'] = ft['taxa'].str.split(' ').str[-1].str[:-1].astype(int)
    ft['sp_short'] = ft['taxa'].str.split(' ').str[0:2]
    
    ft['taxa_order'] = ft.apply(lambda row: tax_order_sp(row, tax_order_dict, sp_dict), axis = 1)
    
    # To keep assignments on GENUS level
    ft['to_TMP'] = ft['taxa_order'].str.split(',').str[5] + ',' + ft['taxa_order'].str.split(',').str[4] + ',' + ft['taxa_order'].str.split(',').str[3] + ',' + ft['taxa_order'].str.split(',').str[2] + ',' + ft['taxa_order'].str.split(',').str[1] + ',' + ft['taxa_order'].str.split(',').str[0]
    
    # Drop columns
    ft.drop(columns= ['taxa_orderTMP', 'sp_short', 'taxa_order'], inplace=True)
    ft.rename(columns={'to_TMP':'taxa_order'}, inplace=True)
    
    # Pair with fastq headers
    ft['read'] = ft.index.map(fastq)
    
    #########################################################################
    #########################################################################
    # Load DL model and encoder
    
    model = keras.models.load_model(sys.argv[4])
    encoder = pickle.load(open(sys.argv[5], 'rb'))
    
    #########################################################################
    #########################################################################
    ########### DL reassignment #############
    
    unassigned_string = ','.join(['Unassigned'] * 6)
    
    dl_l = []
    
    info_coordXY = ft.loc[:,['x','y']]
    info_coordXY.drop_duplicates(inplace=True)
    info_coordXY['tuple'] = list(zip(info_coordXY['x'], info_coordXY['y']))
    
    for tup in info_coordXY['tuple'].tolist():
        
        # Select spot      
        info_xy = ft[(ft['x'] == tup[0]) & (ft['y'] == tup[1])]
         
        if info_xy.shape[0] >0:
            Xpad = stack_padding(info_xy) # Stacking and padding/masking of reads
            y_taxaorder, fastqH, umi = extract_taxa_info(info_xy, 'taxa_order', 'ST') # Collect taxa info
            predictions = predict_taxa_in_model(Xpad, model) # Predict assignments using model
    
            rv_predictions = encoder.inverse_transform(predictions.argmax(axis=1)) # Predict taxa using encoder
            
            # Reassign taxa based on prediciton
            new_taxaorder = []
            
            for i, taxon in enumerate(y_taxaorder):
                if taxon.startswith('Unassigned'):
                    new_taxa = rv_predictions[i]
                    
                    # If unassigned by DL model
                    if new_taxa == '':
                        new_taxa = unassigned_string
                    
                    new_taxaorder.append(new_taxa)
                else:
                    new_taxaorder.append(taxon)
    
            # Store in df
            p = pd.DataFrame({'fastq':fastqH, 'taxa_order':new_taxaorder, 'UMI':umi})
            # Add spot coord
            p['x'] = tup[0].split(':')[-1]
            p['y'] = tup[1].split(':')[-1] 
    
            dl_l.append(p)
    
    df = pd.concat(dl_l)
    
    # Some stats, before dropping unassigned
    stat_file = os.path.join(path, base + "UMIfilt_stats.tsv")
    with open(stat_file, 'a') as f:
        print('Number of reads BEFORE dropping unassigned: ' + str(df.shape[0]), file=f)
    
    # Drop 'Unassigned' *6
    df = df[df['taxa_order'] != unassigned_string]
    
    with open(stat_file, 'a') as f:
        print('Number of reads AFTER dropping unassigned: ' + str(df.shape[0]), file=f)
    
    #########################################################################
    #########################################################################
    ##### Do UMI collapsing with 1 allowed mismatch PER spot AND taxa  ######
    
    # Some stats - before UMI collapsing
    with open(stat_file, 'a') as f:
        print('Number of reads BEFORE UMI collapsing: ' + str(df.shape[0]), file=f)
    
    spot_dfs = []
    
    for spot, group in df.groupby(['x', 'y']):
        
        # All UMIs per spot. Remove the "B3..."
        umi_list = [bytes(item[5:], 'utf-8') for item in group['UMI'].tolist()] 
     
        # First, count number of occurances for each UMI
        umi_raw_counts = dict(Counter(umi_list))
    
        # Run UMI-tools
        clusterer = UMIClusterer(cluster_method="directional")
        umi_groups = clusterer(umi_raw_counts, threshold=1)
        
        # Identify the collapsed UMI per UMI
        group['UMIcollapsed'] = group.apply(lambda row: umi_grouping(row, umi_groups), axis=1)
        
        # Remove rows where taxa and UMI are identical (per spot)
        group = group.drop_duplicates(subset=['taxa_order', 'UMIcollapsed'])
    
        # Add spots (x, y) to group
        group['x'] = spot[0]
        group['y'] = spot[1]
    
        spot_dfs.append(group)
        
    dff = pd.concat(spot_dfs)
    dff.set_index('fastq', inplace=True)
    # Some stats after UMI collapsing
    with open(stat_file, 'a') as f:
        print('Number of reads AFTER UMI collapsing: ' + str(dff.shape[0]), file=f)
        print('\n', file=f)
        print('Reads lost in UMI collapsing: ' + str((df.shape[0] - dff.shape[0])/df.shape[0]) + ' %', file=f)
        
    #########################################################################
    #########################################################################
    # Create an output tsv file
    
    # Spot coordinate xXy
    dff['spot_coord'] = dff['x'].str.split(':').str[-1] + 'x' + dff['y'].str.split(':').str[-1]
    
    # Count taxa per spot coordinate
    
    dff.drop(columns=['x', 'y'], inplace=True)
    dff.reset_index(inplace=True)
    print(dff.head())
    
    tsv = dff.groupby(['spot_coord','taxa_order']).count()[['fastq']].reset_index().pivot(index='spot_coord', columns='taxa_order', values='fastq')
    tsv.fillna(0, inplace = True)
    
    #print to output file
    tsv_file = os.path.join(path, base + "_Bacstdata.tsv")
    
    tsv.to_csv(tsv_file, sep = '\t')


#########################################################################
#########################################################################
def umi_grouping(row, umi_groups):
    """
    Function to define which UMI has been collapsed to which UMI.
    
    Returns the collapsed UMI
    """
    d = {}
    
    umi = bytes(row['UMI'][5:], 'utf-8')

    for umi_group in umi_groups:
        top_umi = umi_group[0]
    
        for raw_umi in umi_group:
            d[raw_umi] = top_umi
        
    return d[umi]

def taxa_assignment(report):
    """
    Create a dictionary with the taxa order with taxID as key and taxa order as value
    Takes the kraken2 report as input and Returns the dictionary.
    """
    tax_order_dict = {}
    sp_tax_dict = {} # This is an extra dictionary with only species and theri corresponding NCBI taxID

    for index, row in report.iterrows():
        
        # Domain
        if row['torder'] == 'D':
            domain = row['sci-name'].lstrip()
            dvalue = ','.join([domain] + ['Unassigned'] * 6)
            tax_order_dict[row['ncbi taxID']] = dvalue
        # Subdomain
        elif row['torder'] == 'D1':
            tax_order_dict[row['ncbi taxID']] = dvalue

        # Phylum
        elif row['torder'] == 'P':
            phylum = row['sci-name'].lstrip()
            pvalue = ','.join([domain] + [phylum] + ['Unassigned'] * 5)
            tax_order_dict[row['ncbi taxID']] = pvalue

        elif ((row['torder'].startswith('P')) & (len(row['torder']) > 1)): # if x1 or x2 or xn
            tax_order_dict[row['ncbi taxID']] = pvalue

        # Class
        elif row['torder'] == 'C':
            Class = row['sci-name'].lstrip()
            cvalue = ','.join([domain] + [phylum] + [Class] + ['Unassigned'] * 4)
            tax_order_dict[row['ncbi taxID']] = cvalue

        # Order
        elif ((row['torder'] == 'O') | (row['torder'] == 'C1')): 
            order = row['sci-name'].lstrip()
            ovalue = ','.join([domain] + [phylum] + [Class] + [order] + ['Unassigned'] * 3)
            tax_order_dict[row['ncbi taxID']] = ovalue            

        # Family
        elif ((row['torder'] == 'F') | (row['torder'] == 'O1') | (row['torder'] == 'C2')):
            family = row['sci-name'].lstrip()
            fvalue = ','.join([domain] + [phylum] + [Class] + [order] + [family] + ['Unassigned'] * 2)
            tax_order_dict[row['ncbi taxID']] = fvalue

        # Genus
        elif ((row['torder'] == 'G') | (row['torder'] == 'F1') | (row['torder'] == 'O2') | (row['torder'] == 'C3')):
            genus = row['sci-name'].lstrip()
            gvalue = ','.join([domain] + [phylum] + [Class] + [order] + [family] + [genus] + ['Unassigned'] * 1)
            tax_order_dict[row['ncbi taxID']] = gvalue

        # species
        elif ((row['torder'] == 'S') | (row['torder'] == 'G1') | (row['torder'] == 'F2') | (row['torder'] == 'O3') | (row['torder'] == 'C4')):
            specie = row['sci-name'].lstrip()
            svalue = ','.join([domain] + [phylum] + [Class] + [order] + [family] + [genus] + [row['sci-name'].lstrip()])
            tax_order_dict[row['ncbi taxID']] = svalue
            sp_tax_dict[specie] = row['ncbi taxID']
        
        # Subspecies    
        elif row['torder'] == 'S1':
            tax_order_dict[row['ncbi taxID']] = svalue

    return tax_order_dict, sp_tax_dict
   
def tax_order_sp(row, d, d_sp):
    """
    Based on taxID, return the taxa order found in d.
    If not taxID found in d (it is a subspecies), look for the genus+species in the d_sp.
    Returns taxa order.
    """
    if row['taxa_orderTMP'] == 10090:
        return 'Animalia,Chordata,Mammalia,Rodentia,Muridae,Mus,Mus musculus'
    
    elif row['taxa_orderTMP'] == 131567: # Cellular organism
        return ','.join(['Cellular organism'] + ['Unassigned'] * 6)
    
    else:
        try:
            taxorder = d[row['taxa_orderTMP']]
            return taxorder
        except KeyError:
            sp = ' '.join([row['sp_short'][0], row['sp_short'][1]])
            
            try:        
                new_taxid = d_sp[sp]
                return d[new_taxid]
            except KeyError:
                return ','.join(['Unassigned'] * 7)
           
def parser(filename):
    """
    Generate a list of tuples (header, read)
    """
    fastq_parsed = {}
    try:

        with open(filename) as fq:
            header = next(fq)
            read = next(fq)
            fastq_parsed[header[1:-1].split(' ')[0]] = read[:-1] # we don't want new-line chars or @ sign in reads
            while True:
                next(fq) # skip + line
                next(fq) # skip qscore
                header = next(fq) # store next read header
                read = next(fq) # store next read seq
                fastq_parsed[header[1:-1].split(' ')[0]] = read[:-1]
    except:
        StopIteration # when we run out of reads, stop
    return fastq_parsed
   
def dna_encode_embedding_table(dna_input, name="dna_encode"):
    """
    DNA embedding.
    """

    embedding_values = np.zeros([len(dna_input), 5], np.float32)
    values = ("A", "C", "G", "T", "N")
    for j, b in enumerate(dna_input):
        if b in values:
            embedding_values[j, values.index(b)] = 1
    return embedding_values

def stack_padding(info_xy):
    # Stack reads into one tenso
    
    info_xy['one hot tensor'] = info_xy.apply(lambda row: dna_encode_embedding_table(row['read']), axis=1)
    X = np.array(info_xy['one hot tensor'].tolist(), dtype=object)

    # Padding to the same sequence length
    masking_value = -1
    max_seq_len = max(len(x) for x in info_xy['one hot tensor'].tolist())
    N = X.shape[0]
    dimension = 5

    Xpad = np.full((N, max_seq_len, dimension), fill_value=masking_value)
    for s, x in enumerate(X):
        seq_len = x.shape[0]
        Xpad[s, 0:seq_len, :] = x
        
    return Xpad

def extract_taxa_info2(info_xy, column_header, slide_type):
    y_taxaorder = info_xy[column_header].tolist()
    y_fastqH = info_xy.index.tolist()

    if slide_type == 'ST':
        y_umi = info_xy['UMI'].tolist()
        return y_taxaorder, y_fastqH, y_umi
    elif slide_type == 'QC':
        return y_taxaorder, y_fastqH

def reassign_classes_per_spot(info, model, encoder):
        
    # Go through all the spots
    clusterl_d = {}
    reassign_D = {}
    
    #for infok, info in info_d.items(): # infok - step or i, info - data frame that comes from different file
    
    cluster_l = []
    reassign_d = {} 
    test_count = 0
    its_n = 0
    unassigned_by_DL = 0
    unassigned_by_DL_tot = 0
    unassigned_by_DL_fastq = []
    
    info_coordXY = info.loc[:,['Bx','By']]
    info_coordXY.drop_duplicates(inplace=True)
    info_coordXY['tuple'] = list(zip(info_coordXY['Bx'], info_coordXY['By'])) # creates a list of tuples with coordinates of barcode x & y
    
    
    for tup in tqdm(info_coordXY['tuple'].tolist()):
        newcluster_d = {}
        spot_coord = str(tup[0]) + 'x' + str(tup[1])  #tup[0].split(':')[-1] + 'x' + tup[1].split(':')[-1]        
    
        info_xy = info[(info['Bx'] == tup[0]) & (info['By'] == tup[1])] # select given spot for x & y
    
        if info_xy.shape[0] > 0:
            Xpad = stack_padding(info_xy) # stacking and padding/masking of reads
    
            taxa_order = info_xy[reversed(['superkingdom', 'phylum', 'class', 'order', 'family', 'genus','species'])]
            taxa_order = taxa_order[[x for x in taxa_order.columns]].apply(lambda x: ','.join(x), axis = 1)
            y_taxaorder = taxa_order.tolist() # collect taxa info

            fastqH = info_xy['fastq'].tolist()
            
            predictions = model.predict(Xpad, verbose = 0) # predict assignments using the model
            rv_predictions = encoder.inverse_transform(predictions.argmax(axis=1)) # predict taxa using encoder
 
            # reassign taxa based on prediciton
            new_taxaorder = []
    
            for i, taxon in enumerate(y_taxaorder): # iterate over kraken output
                if taxon.startswith('unassigned'): # in other words, if spiecies was not classified by Kraken2 swap to DL
                    new_taxa = rv_predictions[i] # when unassigned, choose from DL

                    new_taxaorder.append(new_taxa)
                else:
                    if taxon == rv_predictions[i]: # checking the overlap between Kraken2 and DL
                        unassigned_by_DL += 1 
                        unassigned_by_DL_tot += 1
                        unassigned_by_DL_fastq.append(fastqH[i])
                    else:
                        unassigned_by_DL_tot += 1
    
                    new_taxaorder.append(taxon) # keep the one from kraken2
    
            newcluster_d[spot_coord] = dict(Counter(new_taxaorder))
            reassign_d[spot_coord] = pd.DataFrame({'fastq':fastqH, 'Predicted':new_taxaorder, 'Kraken2':y_taxaorder})

            cluster_l.append(newcluster_d)
            test_count +=1
        #its_n +=1
        #if its_n==10:
        #    break

    print("Common between Kraken2 & DL: " + str(round(unassigned_by_DL/unassigned_by_DL_tot * 100, 2)))
    return cluster_l, reassign_d


def merge_prediction_results(info, cluster_l, fastq_spot_d, taxa_orders, reassign_d):
    # Current assignments
    info2 = info[['fastq','truth_taxa_order']]
    info2.dropna(subset=['truth_taxa_order'], inplace=True)
    
    tx_rv2 = taxa_orders
    tx_rv = tx_rv2[:-1]
    
    # new data frame with split value columns
    new_info = info2["truth_taxa_order"].str.split(",", expand=True)
    new_info = new_info.iloc[:, ::-1]
    new_info.columns = tx_rv 
    
    info3 = pd.concat([info2, new_info], axis=1)
    
    truth_d = info3
    #### Pearsons ####

    #taxa_orders.reverse() #["Doamin", "Phylum", "Class", "Order", "Family", "Genus", "Species"]

    braycurtis_d = {} 
    pearson_d = {}
    accuracy_d = {}
    precision_d = {}
    recall_d = {}
    
    for spot in tqdm(cluster_l): # for every spot, in list of spots cluster_l
        for k,v in spot.items(): # k: XxY, v: {}

            ### Predicted assignments
            predicted = pd.DataFrame.from_dict(v, orient='index', columns=['count'])
            predicted.reset_index(inplace=True)
            predicted = predicted[predicted['index'].notnull()]

            #predicted['index'] = predicted.apply(lambda row: dl_evaluation.rm_specie(row['index']), axis=1) # to remove species
    
            # Separate into taxa levels
            predicted[tx_rv2] = predicted["index"].str.split(pat=",", expand=True)

            ### Truth assignments
            truth_spot = truth_d[truth_d['fastq'].isin(fastq_spot_d[k])]

            ### Kraken2 
            kraken = info.loc[(info['Bx'] == k.split('x')[0]) & (info['By'] == k.split('x')[1])]
            kraken2 = kraken[tx_rv2]
            kraken = pd.concat([kraken['kraken_preds'], kraken2], axis=1)

            predicted_full = reassign_d[k]
            predicted_full.reset_index(inplace=True)
            predicted_full[tx_rv2] = predicted_full["Predicted"].str.split(pat=",", expand=True)
            
            if truth_spot.shape[0] > 0:
    
                for level in taxa_orders[:-1]:
                    
                    pred_taxa = predicted_full[level]
                    krak_taxa = kraken[level]
                    tru_taxa = truth_spot[level]
                    # accuracy
                    acc_dl = metrics.accuracy_score(pred_taxa, tru_taxa)
                    acc_krk = metrics.accuracy_score(krak_taxa, tru_taxa)

                    ppv_dl = metrics.precision_score(pred_taxa, tru_taxa, average='weighted')
                    ppv_krk = metrics.precision_score(krak_taxa, tru_taxa, average='weighted')

                    rec_dl = metrics.recall_score(pred_taxa, tru_taxa, average='weighted')
                    rec_krk = metrics.recall_score(krak_taxa, tru_taxa, average='weighted')
                    
                    acc_d = {}
                    ppv_d = {}
                    rec_d = {}
                    acc_d[level] = [acc_dl, acc_krk]
                    ppv_d[level] = [ppv_dl, ppv_krk]   
                    rec_d[level] = [rec_dl, rec_krk]   
                    if not np.isnan(acc_dl):
                        if not k in pearson_d:
                            accuracy_d[k] = acc_d
                            precision_d[k] = ppv_d
                            recall_d[k] = rec_d
                        else:
                            accuracy_d[k][level] = [acc_dl, acc_krk]
                            precision_d[k][level] = [ppv_dl, ppv_krk]
                            recall_d[k][level] = [rec_dl, rec_krk] 
                            
                    #precision
                    
                    pred = predicted.groupby(level)['count'].sum()
                    tru = truth_spot.groupby(level).count()['truth_taxa_order']
                    krak = kraken.groupby(level).count()['kraken_preds']
                    
                    if 'unassigned' in krak.index: # otherwise kraken2 is lowered down with Pearson
                        krak = krak.drop(['unassigned'])

                    # Comapre relative abundances 
                    bc_d = {}
                    dff = pd.concat([tru, pred, krak], axis=1)
                    
                    dff.columns = ['Truth', 'DL model', 'Kraken2']
                    dff.fillna(0, inplace=True) # to point what tool found 0 taxa

                        # Test to compare relative abundances
                    for col in ['Truth', 'DL model', 'Kraken2']:
                        dff[col + '%'] = (dff[col] / dff[col].sum())
                    
                    tru_pred = distance.braycurtis(dff['Truth%'].tolist(), dff['DL model%'].tolist())
                    tru_krak = distance.braycurtis(dff['Truth%'].tolist(), dff['Kraken2%'].tolist())
                    bc_d[level] = [tru_pred, tru_krak] # B-C truth vs predictied, B-C truth vs kraken2

                    if not k in braycurtis_d:
                        braycurtis_d[k] = bc_d
                    else:
                        braycurtis_d[k][level] = [tru_pred, tru_krak] # B-C truth vs predictied, B-C truth vs kraken2
                        
                    # Check if they are normally distributed
          
                    #res1 = shapiro(dff['Truth%'].tolist())
                    #res2 = shapiro(dff['DL model%'].tolist())
                    #res3 = shapiro(dff['Kraken2%'].tolist())
                    #print(res1)
                    #print(res2)
                    #sprint(res3)
                    # Compare relative abundances with pearson correlation     
                    ps_d = {}  
                    tru_pred = stats.pearsonr(dff['Truth%'].tolist(), dff['DL model%'].tolist())[0] #stats.pearsonr linregress
                    tru_krak = stats.pearsonr(dff['Truth%'].tolist(), dff['Kraken2%'].tolist())[0]
    
                    if not np.isnan(tru_pred):
                        ps_d[level] = [tru_pred, tru_krak] # pearson truth vs predictied, B-C truth vs disrupted
    
                        if not k in pearson_d:
                            pearson_d[k] = ps_d
                        else:
                            pearson_d[k][level] = [tru_pred, tru_krak] # pearson truth vs predictied, eucledian truth vs disrupted            
    
    return pearson_d, braycurtis_d, accuracy_d, precision_d, recall_d


def per_spot_stats(info, reassign_d, fastq_spot_d, taxa_orders):
    
    info2 = info[['fastq','truth_taxa_order']]
    info2.dropna(subset=['truth_taxa_order'], inplace=True)
    tx_rv2 = taxa_orders[:-1] # dropping superkingdom
    
    # new data frame with split value columns
    new_info = info2["truth_taxa_order"].str.split(",", expand=True)
    new_info = new_info.iloc[:, ::-1]
    new_info.columns = tx_rv2
    
    info3 = pd.concat([info2, new_info], axis=1)
    
    truth_d = info3
    
    
    cm_d = {}
    stats_d = {}
    
    for k,v in tqdm(reassign_d.items()):
    
        ### Truth assignments
        truth_spot = truth_d[truth_d['fastq'].isin(fastq_spot_d[k])]
        truth_spot.rename(columns={'index':'fastq'}, inplace=True)
        ### Predicted
        predicted = v[['fastq', 'Predicted']]
        predicted['DL model'] = predicted['Predicted'] #predicted.apply(lambda row: rm_specie(row['Predicted']), axis=1)
        predicted[taxa_orders] = predicted["DL model"].str.split(pat=",", expand=True)
        predicted = predicted.iloc[:, :-1]

        ### Kraken2
        kraken = v[['fastq', 'Kraken2']]
        #kraken['Kraken2'] = kraken.apply(lambda row: rm_specie(row['Kraken2']), axis=1)
        kraken[taxa_orders] = kraken["Kraken2"].str.split(pat=",", expand=True)
        kraken = kraken.iloc[:, :-1]
        tmp_taxa_df = []
        if truth_spot.shape[0] > 0:
    
            for level in tx_rv2:
                pred = predicted[['fastq', level]]
                tru = truth_spot[['fastq', level]]
                kra = kraken[['fastq', level]]
    
                mer = reduce(lambda x,y: pd.merge(x,y, on='fastq', how='outer'), [pred, tru, kra])
                mer.dropna(inplace=True) ###### TO remove when testing to keep 'unclassified' after Kraken2

                # Make confusion matrix
                ## Normalizes confusion matrix over the true (rows), predicted (columns) conditions or all the population.
                cmtmp = {}
    
                cm_tru_pred = confusion_matrix(mer[level + '_y'].tolist(), mer[level + '_x'].tolist(), normalize='true')
                cm_tru_krak = confusion_matrix(mer[level + '_y'].tolist(), mer[level].tolist(), normalize='true')
                
                cmtmp[level] = [cm_tru_pred, cm_tru_krak]
    
                if not k in cm_d:
                    cm_d[k] = cmtmp
                else:
                    cm_d[k][level] = [cm_tru_pred, cm_tru_krak]
    
                    # Calculate stats (unweighted)
                stattmp = {}
    
                stats_tru_pred = classification_report(mer[level + '_y'].tolist(), mer[level + '_x'].tolist(), 
                                              labels=list(set(mer[level + '_y'].tolist())), 
                                              target_names=list(set(mer[level + '_y'].tolist())), 
                                              sample_weight=None, digits=2, 
                                              output_dict=True, 
                                              zero_division='warn')
    
                stats_tru_krak = classification_report(mer[level + '_y'].tolist(), mer[level].tolist(), 
                                              labels=list(set(mer[level + '_y'].tolist())), 
                                              target_names=list(set(mer[level + '_y'].tolist())), 
                                              sample_weight=None, digits=2, 
                                              output_dict=True, 
                                              zero_division='warn')
    
                TP_pred, FN_pred, NA_pred, misclassified_pred = calculate_missclassified(mer[level + '_y'].tolist(), mer[level + '_x'].tolist())
                TP_krak, FN_krak, NA_krak, misclassified_krak = calculate_missclassified(mer[level + '_y'].tolist(), mer[level].tolist())
    
                try:
                    data_tuples = list(zip([level, 'DL model', stats_tru_pred['accuracy'], stats_tru_pred['macro avg']['f1-score'], FN_pred, NA_pred, misclassified_pred],
                                      [level, 'Kraken2', stats_tru_krak['accuracy'], stats_tru_krak['macro avg']['f1-score'], FN_krak, NA_krak, misclassified_krak]))
                    data = [[level, stats_tru_pred['accuracy'], stats_tru_pred['macro avg']['f1-score'], FN_pred, NA_pred, misclassified_pred],
                            [level, stats_tru_krak['accuracy'], stats_tru_krak['macro avg']['f1-score'], FN_krak, NA_krak, misclassified_krak]]

                except KeyError:
                    try:
                        data_tuples = list(zip([level, 'DL model', stats_tru_pred['accuracy'], stats_tru_pred['macro avg']['f1-score'], FN_pred, NA_pred, misclassified_pred],
                                      [level, 'Kraken2', stats_tru_krak['micro avg']['recall'], stats_tru_krak['macro avg']['f1-score'], FN_krak, NA_krak, misclassified_krak]))
                    except KeyError:
                        try:
                            data_tuples = list(zip([level, 'DL model', stats_tru_pred['micro avg']['recall'], stats_tru_pred['macro avg']['f1-score'], FN_pred, NA_pred, misclassified_pred],
                                      [level, 'Kraken2', stats_tru_krak['micro avg']['recall'], stats_tru_krak['macro avg']['f1-score'], FN_krak, NA_krak, misclassified_krak]))
                        except KeyError:
                            data_tuples = list(zip([level, 'DL model', stats_tru_pred['micro avg']['recall'], stats_tru_pred['macro avg']['f1-score'], FN_pred, NA_pred, misclassified_pred],
                                      [level, 'Kraken2', stats_tru_krak['accuracy'], stats_tru_krak['macro avg']['f1-score'], FN_krak, NA_krak, misclassified_krak]))
    
                
                df_stats = pd.DataFrame(data_tuples, index = ['taxa order', 'type','accuracy', 'F1-score', 'False Pos', 'NAs', 'Misclassified']).T
    
                stattmp[level] = df_stats
    
                if not k in stats_d:
                    stats_d[k] = stattmp
                else:
                    stats_d[k][level] = df_stats

    return stats_d
