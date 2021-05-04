import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    prefix='/Volumes/T7 Touch/tmp/resultsData/'
    eps=0.001
    geo=1 # use 0/1 
    maxval_range=np.arange(3,5+eps,0.5)
    # logslope_range=np.arange(3.5, 4.5+eps,1)
    utility='logHm';
    meanValues = np.arange(-1,3+eps,0.1)

    if geo == True:
        geoTitle = 'nonlinearTime'
        cost='g-0.8'
        logslope_range=np.arange(3.5, 6.5+eps,1)
        ylimit=50
        ticksize=6
        titlesize=8
        yvalues = [10,25,40]
    else:
        geoTitle = 'linearTime'
        cost='c-0.6'
        logslope_range=np.arange(0.25, 1.5+eps,0.25)
        ylimit=0.15
        ticksize=5
        titlesize=6
        yvalues = [0.03,0.08,0.13]
    
    suffix='_rm-1.5_S-8-151_'+cost+'_t-50-multiDec_u-'
    
    fig, axs = plt.subplots(len(maxval_range), len(logslope_range), sharex='col', sharey='row',
                        gridspec_kw={'hspace': 0, 'wspace': 0})
    # plt.setp(axs.flat, xlabel='X-label', ylabel='Y-label')
    xvalues = np.arange(-0.5,2.5+eps,1.5)
    # yvalues = np.arange(0,50+eps,10)
    
    row=0
    for maxval in maxval_range:
        col=0
        for logslope in logslope_range:
            maxvalstr = str(maxval) if maxval%1>0 else str(int(maxval))
            logslopestr = str(logslope) if logslope%1>0 else str(int(logslope))
            filename = prefix + 'vs_geometric-' + str(geo) + '_mv-' + maxvalstr + '_ls-' + logslopestr + suffix + utility + '.txt';
            print("Reading file " + filename + " ...")
            
            data = pd.read_csv(filename,sep=',',header=None)
            data = pd.DataFrame(data)
            
            print('Loaded file with ' + str(len(data)/len(meanValues)) + ' lines.');
            means = []
            errorsP = []
            errorsM = []
            for meanValue in meanValues:
                selectedData = data[(data[0] - meanValue < 0.000001)][2]
                means.append(np.mean(selectedData))
                err = 1.96 * np.std(selectedData)/np.sqrt(len(selectedData))
                errorsP.append( np.mean(selectedData) + err )
                errorsM.append( np.mean(selectedData) - err )
            
            # plt.plot(meanValues, means,'k-')
            # plt.fill_between(meanValues, errorsP, errorsM, alpha=0.2)
            # plt.show()
            axs[row,col].plot(meanValues, means,'k-')
            axs[row,col].fill_between(meanValues, errorsP, errorsM, alpha=0.5, color='r')
            axs[row,col].set_ylim([0, ylimit])
            
            #axs[row,col].xaxis.set_tick_params(labelsize=7)
            #axs[row,col].yaxis.set_tick_params(labelsize=7)
            # axs[row,col].set_xlabel("stimuli's magnitude")
            
            # axs[row,col].set_xticklabels([-1,1,3], fontsize=7)
            # axs[row,col].set_yticklabels([0,25,50], fontsize=7)
            # plt.xticks(fontsize=7)
            # plt.yticks(fontsize=7)
            col=col+1
        row=row+1
    
    pad = 5 # in points
    for ax, ls in zip(axs[0], logslope_range):
        #ax.set_title("logslope = " +str(ls), fontsize=7 )
        ax.annotate("logslope = " +str(ls), xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline', fontsize=titlesize)
    
    for ax in axs[-1]:
        ax.set_xlabel("stimuli's magnitude", fontsize=ticksize)
        ax.set_xticks(xvalues)
        ax.xaxis.set_tick_params(labelsize=ticksize)

    for ax, mv in zip(axs[:,0], maxval_range):
        ax.set_ylabel("reaction time", rotation=90, fontsize=ticksize)
        ax.set_yticks(yvalues)
        ax.yaxis.set_tick_params(labelsize=ticksize)
        ax.annotate("mv = " +str(mv), xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center', fontsize=titlesize)

    #plt.show()
    fig.savefig("multiplot_" + geoTitle + "_" + cost + "_" + utility + ".pdf", bbox_inches='tight')


