prefix='/Users/marshall/Google Drive/My Drive/DiODe/Data/MultiAlternativeDecisions/resultsData/';
%prefix='/Users/marshall/Google Drive/My Drive/Home/Git/MultiAlternativeDecisions/non-linear-time/resultsData/';
figure;
%geo=0;
geo=1;
maxval_range=3:0.5:5;
%logslope_range=0.25:0.25:1.5;
logslope_range=2.5:0.5:5;
tmax=3;
%tmax=37;
% maxval_range=3;
% logslope_range=0.25;
utility='logHm';
%utility='linear';
meanValues = -3:0.1:3;

if geo == true
    geoTitle = 'nonlinear time';
    cost='g-0.1';
    deci='-multiDec_u-';
else
    geoTitle = 'linear time';
    cost='c-0';
    deci='-multiDec_u-';
end
suffix=strcat('_rm-1.5_S-8-151_',cost,'_t-',int2str(tmax),deci);
%strrep(strrep(utility,'logLm','nonlinear (logistic)'),'sqrt','nonlinear (sqrt)')
row=0;
for maxval=maxval_range
    row=row+1;
    col=0;
    for logslope=logslope_range
        col=col+1;
        filename = strcat(prefix,'vs_geometric-',num2str(geo),'_mv-',num2str(maxval),'_ls-',num2str(logslope),suffix,utility,'.txt');
        try
            allResults = readtable(filename);            
            fprintf('Loaded file %s with %d lines.\n', filename, height(allResults)/length(meanValues));            
            dataForPlot = [];
            for meanValue = meanValues
                dataForPlot = [ dataForPlot, allResults{ abs(allResults{:,1} - meanValue)<0.0000001, 3}];
            end
            subplot(length(maxval_range),length(logslope_range),(row-1)*length(logslope_range)+col); errorbar(meanValues, mean(dataForPlot), std(dataForPlot)/sqrt(height(allResults)/length(meanValues))*1.96,'LineWidth', 1.5);
            title(strcat('s=',num2str(logslope),' m=',num2str(maxval)));
            set(gca,'FontSize',13)
            % scaling the axes
            ymin = min(1, min(mean(dataForPlot)-std(dataForPlot)/sqrt(height(allResults)/length(meanValues))*3) );
            ymax = max(0, max(mean(dataForPlot)+std(dataForPlot)/sqrt(height(allResults)/length(meanValues))*3) );
            %fprintf('%f \n',ymax);
            % Fixing the axes
            ymin=0; if geo == true; ymax=3; else; ymax=0.5; end
            epsilon=(max(meanValues)-min(meanValues))*0.05;
            if ymin==ymax; ymax=ymin+0.0001; end
            axis([min(meanValues)-epsilon max(meanValues)+epsilon ymin ymax]);
        catch foo
            fprintf('Could not load file %s.\n', filename);            
        end
    end
end
%figure()
%errorbar(meanValues, mean(dataForPlot), std(dataForPlot)/sqrt(height(allResults)/length(meanValues))*1.96,'LineWidth', 1.5);


