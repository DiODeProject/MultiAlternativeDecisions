function valueDecisionBoundaryRR2_3D()
global gamm geometric epsil pie; % (JARM 23rd August '19)
global valscale; % (JARM 7th October '19)
geometric = true; % (JARM 23rd August '19) use geometric discounting for future rewards 
gamm = 0.6; % (JARM 23rd August '19) geometric discount factor for future rewards 
gamm = 0.01; % Original value
epsil = 0; % (JARM 11th September '19) epsilon error to add to co-planar services to compute convex hull (required to check geometric discounting results; deprecated)
pie = 0; % (JARM 27th May '20) input-dependent noise scaling 
valscale = 0.5; % (JARM 7th October '19) move triangle along diagonal as option values scale)
maxval = 4; % (JARM 6th March '20) maximum utility for logistic utility function
utility='linear';
tic;
Smax = 8;      % Grid range of states space (now we assume: S = [(Rhat1+Rhat2)/2, (Rhat1-Rhat2)/2]); Rhat(t) = (varR*X(t)+varX)/(t*varR+varX) )
Smax = 4;      % Original value
resSL  = 15;      % Grid resolution of state space
resS = 151;      % Grid resolution of state space
resS = 101;      % Original value
tmax = 37;       % Time limit
tmax = 3;       % Original value
dt   = .0625;      % Time step for computing the decision matrix
dt   = .05;       % Original value
dtsim   = .005;    % Time step for computing the stochastic simulations
c    = 0.1;       % Cost of evidence accumulation
tNull = .25;     % Non-decision time + inter trial interval
g{1}.meanR = 0; % Prior mean of state (dimension 1)
g{1}.varR  = 5; % Prior variance of stte
g{1}.varX  = 2; % Observation noise variance
g{2}.meanR = 0; % Prior mean of state (dimension 2)
g{2}.varR  = 5; % Prior variance of state
g{2}.varX  = 2; % Observation noise variance
g{3}.meanR = 0; % Prior mean of state (dimension 3)
g{3}.varR  = 5; % Prior variance of state
g{3}.varX  = 2; % Observation noise variance
t = 0:dt:tmax;
Slabel = {'r_1^{hat}', 'r_2^{hat}', 'r_3^{hat}'};
myCol = [1 0 0; 0 1 0; 0 0 1];

% Utililty function:
if contains(utility , 'linear')
    utilityFunc = @(X) X;
elseif contains(utility , 'tan')
    utilityFunc = @(X) tanh(X);
elseif contains(utility , 'logisticL')
    logslope = 0.75; % slope parameter for logistic utility function
    utilityFunc = @(X) maxval./(1+exp(-logslope*(X)));
elseif contains(utility , 'logLm')
    logslope = 0.75; % slope parameter for logistic utility function
    utilityFunc = @(X) -maxval + maxval*2./(1+exp(-logslope*(X)));
elseif contains(utility , 'logHm')
    logslope = 1.5; % slope parameter for logistic utility function
    utilityFunc = @(X) -maxval + maxval*2./(1+exp(-logslope*(X)));
elseif contains(utility , 'logisticH')
    logslope = 1.5; % slope parameter for logistic utility function
    utilityFunc = @(X) maxval./(1+exp(-logslope*(X)));
else
    utilityFunc = @(X) sign(X).*abs(X).^0.5;
end

figure;
x=(-Smax:0.1:Smax);
plot(x,utilityFunc(x));
saveas(gcf,['simFigs/' utility '.pdf'])

%% Reward rate, Average-adjusted value, Decision (finding solution):
SscaleL  = linspace(-Smax, Smax, resSL);
[S{1},S{2},S{3}] = ndgrid(SscaleL, SscaleL, SscaleL);
iS0 = [findnearest(g{1}.meanR, SscaleL) findnearest(g{2}.meanR, SscaleL) findnearest(g{3}.meanR, SscaleL)];
for iC = 3:-1:1;  Rh{iC} = utilityFunc(S{iC});  end                                                          % Expected reward for option iC
[RhMax, Dd] = max_({Rh{1}, Rh{2}, Rh{3}});                                                                   % Expected reward for decision
[V0, V, D, EVnext, rho, Ptrans, iStrans] = backwardInduction(g{1}.meanR,c,tNull,g,Rh,S,t,dt,iS0);
if geometric == false
    rho_ = fzero(@(rho) backwardInduction(rho,c,tNull,g,Rh,S,t,dt,iS0), g{1}.meanR)                            % Reward rate
else
    rho_ = 0; % (JARM 20th September '19) reward rate optimisation does not currently converge for geometric discounting
    %rho_ = fzero(@(rho) backwardInduction(rho,c,tNull,g,Rh,S,t,dt,iS0), g{1}.meanR)
end

%% Reward rate, Average-adjusted value, Decision (high resolution):
Sscale = linspace(-Smax, Smax, resS);
[S{1},S{2},S{3}] = ndgrid(Sscale, Sscale, Sscale);
iS0 = [findnearest(g{1}.meanR, Sscale) findnearest(g{2}.meanR, Sscale) findnearest(g{3}.meanR, Sscale)];
for iC = 3:-1:1;  Rh{iC} = utilityFunc(S{iC});  end                                                          % Expected reward for option iC
[RhMax, Dd] = max_({Rh{1}, Rh{2}, Rh{3}});                                                                   % Expected reward for decision
[V0, V, D, EVnext, rho, Ptrans, iStrans] = backwardInduction(rho_,c,tNull,g,Rh,S,t,dt,iS0);                  % Average-adjusted value, Decision, Transition prob. etc.

%% Transform to the space of accumulated evidence:
% dbX = transformDecBound(dbS2,Sscale,t,g);

%% Run set of tests to measure value-sensitivity in equal alternative case
for utility = ["linear", "tan", "logLm", "logHm", "sqrt"] 
for geo = [false true]
%for geo = false
if geo; geometric=true; else; geometric=false; end
savePlots = false;
updateMeanVarianceEachStep=false; % if true, it uses the posterior as prior mean and variance in the next dt observation
computeDecisionBoundaries=false; % if true the code will compute the decision boundaries, if false will try to load the decision boundaries from the directory rawData (if it fails, it will recompute the data)
priorMeanEqualToEvidenceDataMean=false; % if true the prior mean corresponds exactly to the used evidence data mean, if false the prior mean is a fixed value
singleDecisions=true; % if true we model single decision (i.e. expected future rewards=0); if false we compute the expected future reward (rho_)
meanValues=0.5:0.5:3; % mean reward values to be tested 
%meanValues = 0:1:3; % mean reward values to be tested 
numruns=1000; % number of simulations per test case
allResults=zeros(length(meanValues)*numruns, 3); % structure to store the simulation results
j=1; % result line counter

% prepare file suffix
if priorMeanEqualToEvidenceDataMean
    priorMeanSuffix='-varP';
else
    priorMeanSuffix='-fixP';
end
if singleDecisions
    singleDecisionsSuffix='-singleDec';
else
    singleDecisionsSuffix='-multiDec';
end
if geometric
    cost=gamm;
else
    cost=c;
end
suffix=[priorMeanSuffix singleDecisionsSuffix '_u-' utility];

Sscale = linspace(-Smax, Smax, resS); % define the range of possible rewards
[S{1},S{2},S{3}] = ndgrid(Sscale, Sscale, Sscale); % define the space of possible rewards
Rh{1} = utilityFunc(S{1}); % Expected reward for option 1
Rh{2} = utilityFunc(S{2}); % Expected reward for option 2
Rh{3} = utilityFunc(S{3}); % Expected reward for option 3
for meanValue = meanValues
    if priorMeanEqualToEvidenceDataMean
        g{1}.meanR = meanValue; g{2}.meanR = meanValue; g{3}.meanR = meanValue; % set prior with equal true mean for all three options
    else
        fixedMeanValue = 1.5;
        g{1}.meanR = fixedMeanValue; g{2}.meanR = fixedMeanValue; g{3}.meanR = fixedMeanValue; % fix prior of mean to a fixed value 
    end
    option1Mean = meanValue; option2Mean = meanValue; option3Mean = meanValue; % set actual mean, from which the evidence data is drawn, equal for both options
    if geometric
        filename = strcat('rawData/D_geom-',num2str(geometric),'_rm-',num2str(g{1}.meanR),'_S-',num2str(Smax),'-',num2str(resS),'_g-',num2str(gamm),'_t-',num2str(tmax),singleDecisionsSuffix,'_u-',utility,'.mat');
    else
        filename = strcat('rawData/D_geom-',num2str(geometric),'_rm-',num2str(g{1}.meanR),'_S-',num2str(Smax),'-',num2str(resS),'_c-',num2str(c),'_t-',num2str(tmax),singleDecisionsSuffix,'_u-',utility,'.mat');
    end
    dataLoaded = false;
    if priorMeanEqualToEvidenceDataMean || meanValue==meanValues(1) % skip loading/computing the decision matrix after the first loop if the prior remains the same (i.e. priorMeanEqualToEvidenceDataMean==false)
        if ~computeDecisionBoundaries % load the decision threshold for all timesteps (matrix D)
            try
                fprintf('loading boundaries...');
                load(filename, 'D','rho_');
                dataLoaded = true;
                fprintf('done.\n');
            catch
                disp('Could not load the decision matrix. Recomputing the data.'); 
            end
        end
        if ~dataLoaded % compute the decision threshold for all timesteps
            fprintf('computing boundaries...');
            if singleDecisions
                rho_ = 0; % we assume single decisions
            else
                SscaleL  = linspace(-Smax, Smax, resSL);
                [SLow{1},SLow{2},SLow{3}] = ndgrid(SscaleL, SscaleL, SscaleL);
                iS0 = [findnearest(g{1}.meanR, SscaleL) findnearest(g{2}.meanR, SscaleL) findnearest(g{3}.meanR, SscaleL)]; % this line is not really needed
                for iC = 3:-1:1;  RhLow{iC} = utilityFunc(SLow{iC});  end  % Expected reward for option iC
                rho_ = fzero(@(rho) backwardInduction(rho,c,tNull,g,RhLow,SLow,t,dt,iS0), g{1}.meanR); % compute future rewards decisions
                fprintf('rho for mean %d is %d...',g{1}.meanR, rho_);
            end

            iS0 = [findnearest(g{1}.meanR, Sscale) findnearest(g{2}.meanR, Sscale) findnearest(g{3}.meanR, Sscale)]; % this line is not really needed
            [V0, V, D, EVnext, rho, Ptrans, iStrans] = backwardInduction(rho_,c,tNull,g,Rh,S,t,dt,iS0);
            fprintf('saving boundaries to file...');
            save(filename,'D','rho_', '-v7.3');
            fprintf('done.\n');
        end
    end
    if savePlots % prepare the figure
        figure(); clf;
        set(gcf, 'PaperUnits', 'inches');
        set(gcf, 'PaperSize', [8 1+numruns*1.5]);
        set(gcf, 'PaperPositionMode', 'manual');
        set(gcf, 'PaperPosition', [0 0 8 1+numruns*1.5]);
    end
    for run = 1:numruns
        r1sum=0;
        r2sum=0;
        r3sum=0;
        if updateMeanVarianceEachStep
            r1m = g{1}.meanR; % mean opt 1
            r2m = g{2}.meanR; % mean opt 2
            r3m = g{3}.meanR; % mean opt 3
            r1v = g{1}.varR; % var opt 1
            r2v = g{2}.varR; % var opt 2
            r3v = g{3}.varR; % var opt 3
        end
        simTraj = [ ];
        for simT = 0:dtsim:tmax %iT = 1:1:length(t)-1
            iT = findnearest(simT, t);
            if updateMeanVarianceEachStep
                r1v = (r1v * g{1}.varX) / ( g{1}.varX + t(iT) * r1v ); % compute the posterior variance for op 1
                r2v = (r2v * g{2}.varX) / ( g{2}.varX + t(iT) * r2v ); % compute the posterior variance for op 2
                r3v = (r3v * g{3}.varX) / ( g{3}.varX + t(iT) * r3v ); % compute the posterior variance for op 3
                r1m = r1v * ( (r1m/r1v) + (r1sum/g{1}.varX) ); % compute the posterior mean 1
                r2m = r2v * ( (r2m/r2v) + (r2sum/g{2}.varX) ); % compute the posterior mean 2
                r3m = r3v * ( (r3m/r2v) + (r3sum/g{3}.varX) ); % compute the posterior mean 3
            else
                r1m = ( (g{1}.meanR * g{1}.varX) + (r1sum * g{1}.varR) ) / (g{1}.varX + t(iT) * g{1}.varR ); % compute the posterior mean 1
                r2m = ( (g{2}.meanR * g{2}.varX) + (r2sum * g{2}.varR) ) / (g{2}.varX + t(iT) * g{2}.varR ); % compute the posterior mean 2
                r3m = ( (g{3}.meanR * g{3}.varX) + (r3sum * g{3}.varR) ) / (g{3}.varX + t(iT) * g{3}.varR ); % compute the posterior mean 3
            end
            if savePlots 
                simTraj = [simTraj; r1m r2m r3m ];
            end
            % find the index of the posterior mean to check in matrix D if decision is made
            r1i = findnearest(r1m, Sscale, -1);
            r2i = findnearest(r2m, Sscale, -1);
            r3i = findnearest(r3m, Sscale, -1);
            % fprintf('For values (%d,%d) at time %d the D is %d\n',r1v,r2v, t(iT), D(r1i,r2i,iT) )
            try
                decisionMade = D(r1i,r2i,r3i,iT) ~= 4;
            catch
                fprintf('ERROR for D=%d, t=%d, r1m=%d, r2m=%d, r3m=%d, r1i=%d, r2i=%d, r3i=%d\n',D(r1i,r2i,iT),iT,r1m,r2m,r3m,r1i,r2i,r3i);
            end
            if decisionMade
                break
            else
                r1sum = r1sum + normrnd(option1Mean*dtsim, sqrt(g{1}.varX*dtsim)+pie*(r1sum^2));
                r2sum = r2sum + normrnd(option2Mean*dtsim, sqrt(g{2}.varX*dtsim)+pie*(r2sum^2));
                r3sum = r3sum + normrnd(option3Mean*dtsim, sqrt(g{3}.varX*dtsim)+pie*(r3sum^2));
            end
        end
        if savePlots
            valscale = (r1m + r2m + r3m + Smax)/3; % moving to the correct 2d projection plane
%             dbIS = plotDecisionVolume(S, D(:,:,:,iT), [-Smax Smax] );
%             hold on; plot3(simTraj(:,1),simTraj(:,2),simTraj(:,3),'k','linewidth',2);
%             hold on; plot3(r1v,r2v,r3v,'ko','linewidth',2);
%             figure();
            %clf;
            dbIS = compute3dBoundaries(S, D(:,:,:,iT)); % computing the projected boundaries on 2d plane
            subplot(ceil(numruns/2),2,run); % select subplot
            plotTrajOnProjection(dbIS, simTraj, myCol); % plot the trajectories on the 2d projection
            title(['time=' num2str(t(iT)) ' -- \Sigma_i(r_i)=' num2str(sum(simTraj(iT,1:3))) ]);
            filename = strcat('rawData/traj_geometric-',num2str(geometric),'_r1-',num2str(option1Mean),'_r2-',num2str(option2Mean),'_r3-',num2str(option3Mean),'_',num2str(run),'.txt');
            csvwrite(filename,simTraj);
        end
        dec=D(r1i,r2i,r3i,iT);
        if isempty(dec); dec=0; end
        allResults(j,:) = [ meanValue dec t(iT) ];
        j = j+1;
    end
    if savePlots
        filename = strcat('simFigs/geometric-',num2str(geometric),'_pm-',num2str(g{1}.meanR),'_rm-',num2str(meanValue),'.pdf');
        saveas(gcf,filename)
    end
end
filename = strcat('resultsData/vs_geometric-',num2str(geometric),strjoin(suffix,''),'_c-',num2str(cost),'.txt');
csvwrite(filename,allResults);
end
end

%% Plot value sensitive comparison
for utility = ["linear", "tan", "logLm", "logHm", "sqrt"]
addpath('../plottinglibs/');
suffix1=strcat('-fixP-singleDec_u-',utility,'_c-'); %suffix='-varP';
fprintf("%s\n",suffix1);
figure();
clf;
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [8 4]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 8 4]);
for geo = [true false]
    if geo; cost=gamm; else; cost=c; end
    suffix=strjoin([suffix1 num2str(cost)],'');
    filename = strcat('resultsData/vs_geometric-',num2str(geo),suffix,'.txt');
    allResults = readtable(filename);
    fprintf('Loaded file %s with %d lines.\n', filename, height(allResults)/length(meanValues));
    dataForPlot = [];
    for meanValue = meanValues
        dataForPlot = [ dataForPlot, allResults{ allResults{:,1} == meanValue, 3}];
    end
    if geo == true
        geoTitle = 'geom. cost ';
    else
        geoTitle = 'linear cost ';
    end
    if contains(suffix, '-fixP')
        priorTitle = '- constant prior -';
    else
        priorTitle = '- variable prior -';
    end
    subplot(2,2,1+geo); errorbar(meanValues, mean(dataForPlot), std(dataForPlot)/sqrt(height(allResults)/length(meanValues))*1.96); title(['3n - ' geoTitle priorTitle ' mean']);
    %violin(dataForPlot,'xlabel',string(meanValues))
    subplot(2,2,3+geo); distributionPlot(dataForPlot,'histOpt',1,'colormap',spring,'xValues',meanValues); title(['3n - ' geoTitle priorTitle ' all data']);
end
filename = strcat('simFigs/value-sensitive',strrep(suffix,'.',''),'.pdf');
saveas(gcf,filename)

figure();
clf;
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [8 4]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 8 4]);
ymin=1000; ymax=0;
for geo = [false true]
    if geo; cost=gamm; else; cost=c; end
    suffix=strjoin([suffix1 num2str(cost)],'');
    filename = strcat('resultsData/vs_geometric-',num2str(geo),suffix,'.txt');
    allResults = readtable(filename);
    fprintf('Loaded file %s with %d lines.\n', filename, height(allResults)/length(meanValues));
    dataForPlot = [];
    for meanValue = meanValues
        dataForPlot = [ dataForPlot, allResults{ allResults{:,1} == meanValue, 3}];
    end
    hold on; errorbar(meanValues, mean(dataForPlot), std(dataForPlot)/sqrt(height(allResults)/length(meanValues))*1.96,'LineWidth', 3);
    ymin = min(ymin, min(mean(dataForPlot)-std(dataForPlot)/sqrt(height(allResults)/length(meanValues))*3) );
    ymax = max(ymax, max(mean(dataForPlot)+std(dataForPlot)/sqrt(height(allResults)/length(meanValues))*3) );
end
epsilon=(max(meanValues)-min(meanValues))*0.05;
axis([min(meanValues)-epsilon max(meanValues)+epsilon ymin ymax]);
legend('Linear cost','Geometric cost','FontSize',13)%,'Location','east')
xlabel('Stimuli''s magnitude')
ylabel('Reaction time')
set(gca,'FontSize',13)
pbaspect([3 2 2])
filename = strcat('simFigs/vs-comp',strrep(suffix,'.',''),'.pdf');
saveas(gcf,filename);
end

%% Plot all in one RT-value
figure();
clf;
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [8 4]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 8 4]);
ymin=1000; ymax=0;
for utility = ["linear", "tan", "logLm", "logHm", "sqrt"]
    suffix1=['-fixP-singleDec_u-' utility '_c-'];
    if contains(utility , 'linear'); col='k';
    elseif contains(utility , 'tan'); col='b';
    elseif contains(utility , 'logLm'); col='g';
    elseif contains(utility , 'logHm'); col='r';
    elseif contains(utility , 'sqrt'); col='m'; end
    for geo = [false true]
        if geo; cost=gamm; else; cost=c; end
        if geo; lstyle='-'; else; lstyle='--'; end
        suffix=[suffix1 num2str(cost)];
        filename = strcat('resultsData/vs_geometric-',num2str(geo),strjoin(suffix,''),'.txt');
        allResults = readtable(filename);
        fprintf('Loaded file %s with %d lines.\n', filename, height(allResults)/length(meanValues));
        dataForPlot = [];
        for meanValue = meanValues
            dataForPlot = [ dataForPlot, allResults{ allResults{:,1} == meanValue, 3}];
        end
        hold on; errorbar(meanValues, mean(dataForPlot), std(dataForPlot)/sqrt(height(allResults)/length(meanValues))*1.96,'LineWidth', 3, 'Linestyle', lstyle, 'Color', col);
        ymin = min(ymin, min(mean(dataForPlot)-std(dataForPlot)/sqrt(height(allResults)/length(meanValues))*3) );
        ymax = max(ymax, max(mean(dataForPlot)+std(dataForPlot)/sqrt(height(allResults)/length(meanValues))*3) );
    end
end
epsilon=(max(meanValues)-min(meanValues))*0.05;
axis([min(meanValues)-epsilon max(meanValues)+epsilon ymin ymax]);
h = zeros(5, 1);
h(1) = plot(NaN,NaN,'ok','MarkerFaceColor','k');
h(2) = plot(NaN,NaN,'ob','MarkerFaceColor','b');
h(3) = plot(NaN,NaN,'og','MarkerFaceColor','g');
h(4) = plot(NaN,NaN,'or','MarkerFaceColor','r');
h(5) = plot(NaN,NaN,'om','MarkerFaceColor','m');
legend(h, "linear", "tan", "low logistic", "high logistic", "sqrt",'FontSize',9,'Orientation','horizontal','Position',[0.48,0.8,0.1,0.05]);
xlabel('Stimuli''s magnitude')
ylabel('Reaction time')
set(gca,'FontSize',16)
pbaspect([3 2 2])
filename = 'simFigs/vs-all-in-one.pdf';
saveas(gcf,filename);

%% Plot value sensitive RT 
addpath('../plottinglibs/');
geo=false;
geo=true;
if geo; cost=gamm; else; cost=c; end
suffix=['-fixP-singleDec_u-sqrt_c-' num2str(cost)]; %suffix='-varP';
figure();
clf;
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [8 4]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 8 4]);
filename = strcat('resultsData/vs_geometric-',num2str(geo),suffix,'.txt');
allResults = readtable(filename);
fprintf('Loaded file %s with %d lines.\n', filename, height(allResults)/length(meanValues));
dataForPlot = [];
for meanValue = meanValues
    dataForPlot = [ dataForPlot, allResults{ allResults{:,1} == meanValue, 3}];
end
if geo == true
    geoTitle = 'geom. cost ';
else
    geoTitle = 'linear cost ';
end
if contains(suffix, '-fixP')
    priorTitle = '- constant prior -';
else
    priorTitle = '- variable prior -';
end
errorbar(meanValues, mean(dataForPlot), std(dataForPlot)/sqrt(height(allResults)/length(meanValues))*1.96,'LineWidth', 3);
%title(['3n - ' geoTitle priorTitle ' mean']);
epsilon=(max(meanValues)-min(meanValues))*0.05;
axis([min(meanValues)-epsilon max(meanValues)+epsilon min(mean(dataForPlot)-std(dataForPlot)/sqrt(height(allResults)/length(meanValues))*3) max(mean(dataForPlot)+std(dataForPlot)/sqrt(height(allResults)/length(meanValues))*3)]);
xlabel('Stimuli''s magnitude')
ylabel('Reaction time')
set(gca,'FontSize',13)
pbaspect([3 2 2])
%violin(dataForPlot,'xlabel',string(meanValues))
%distributionPlot(dataForPlot,'histOpt',1,'colormap',bone,'xValues',meanValues); title(['3n - ' geoTitle priorTitle ' all data']);
filename = strcat('simFigs/vs-g',num2str(geo),strrep(suffix,'.',''),'.pdf');
saveas(gcf,filename);
%system(['/Library/TeX/texbin/pdfcrop ' filename ' ' filename ]);

%% Plot boundaries for varying time and magnitude
for utility = ["linear"]%, "tan", "logLm", "logHm", "sqrt"] % WARNING ** This loop works only for plotting NOT for computing
magnitudePlot=true;
computeDecisionBoundaries=false; % if true the code will compute the decision boundaries, if false will try to load the decision boundaries from the directory rawData (if it fails, it will recompute the data)
singleDecisions=true; % if true we model single decision (i.e. expected future rewards=0); if false we compute the expected future reward (rho_)
meanValues=-0.5:1:1.5; % mean reward values to be tested 
timeSnaps=0:0.333:1; % mean reward values to be tested
timeSnaps=0:0.25:0.75;
%timeSnaps=0:0.25:0.;
if contains( ['tan','logHm'],utility )
    %timeSnaps=0:0.1:0.3; % mean reward values to be tested are different
    timeSnaps=0:0.25:0.75; % mean reward values to be tested are different
elseif contains( 'logLm',utility )
    %timeSnaps=0:0.2:0.6; % mean reward values to be tested are different
    timeSnaps=0:0.25:0.75; % mean reward values to be tested are different
end
%meanValues=-0.5:0.5:0.5; 
%meanValues=1.5;
%timeSnaps=0.3;

% prepare plot
figure(); clf;
set(gcf, 'PaperUnits', 'inches');
if magnitudePlot
    set(gcf, 'PaperSize', [1+length(timeSnaps)*3 6]);
else
    set(gcf, 'PaperSize', [1+length(timeSnaps)*1.5 1+length(meanValues)*1]);
end
set(gcf, 'PaperPositionMode', 'manual');
if magnitudePlot
    set(gcf, 'PaperPosition', [0 0 1+length(timeSnaps)*3 6]);
else
    set(gcf, 'PaperPosition', [0 0 1+length(timeSnaps)*1.5 1+length(meanValues)*1]);
end

for geo = [false true]
%for geo = false
if geo; geometric=true; else; geometric=false; end
priorMeanSuffix='-fixP';
if singleDecisions
    singleDecisionsSuffix='-singleDec';
else
    singleDecisionsSuffix='-multiDec';
end
if geometric
    cost=gamm;
else
    cost=c;
end
suffix=[priorMeanSuffix singleDecisionsSuffix '_u-' utility];

Sscale = linspace(-Smax, Smax, resS); % define the range of possible rewards
[S{1},S{2},S{3}] = ndgrid(Sscale, Sscale, Sscale); % define the space of possible rewards
Rh{1} = utilityFunc(S{1}); % Expected reward for option 1
Rh{2} = utilityFunc(S{2}); % Expected reward for option 2
Rh{3} = utilityFunc(S{3}); % Expected reward for option 3
boundariesDataLoaded = false; % flag for loading files
valk=0;
for meanValue = meanValues
    valk=valk+1;
    fixedMeanValue = 1.5;
    g{1}.meanR = fixedMeanValue; g{2}.meanR = fixedMeanValue; g{3}.meanR = fixedMeanValue; % fix prior of mean to a fixed value
    valscale = (3*meanValue + Smax)/3; % moving to the correct 2d projection plane
    timetk=0;
    if magnitudePlot || valk==1 
        j=1 + length(timeSnaps)*(geo&&magnitudePlot); % subplot counter
    end
    for timeSnap = timeSnaps
        iT=findnearest(timeSnap,t);
        if geometric
            filename = strcat('rawData/plane_geom-',num2str(geometric),'_rm-',num2str(g{1}.meanR),'_S-',num2str(Smax),'-',num2str(resS),'_g-',num2str(gamm),'_t-',num2str(tmax),singleDecisionsSuffix,'_u-',utility,'_plane-',num2str(meanValue),'_snap-',num2str(iT),'.mat');
        else
            filename = strcat('rawData/plane_geom-',num2str(geometric),'_rm-',num2str(g{1}.meanR),'_S-',num2str(Smax),'-',num2str(resS),'_c-',num2str(c),'_t-',num2str(tmax),singleDecisionsSuffix,'_u-',utility,'_plane-',num2str(meanValue),'_snap-',num2str(iT),'.mat');
        end
        % try to load the file
        dataLoaded = false;
        if ~computeDecisionBoundaries % load the projected boundaries
            try
                fprintf('loading projection...');
                load(filename, 'dbIS');
                dataLoaded = true;
                fprintf('done.\n');
            catch
                disp('Could not load the projected boundaries. Recomputing the data.');
            end
        end
        if ~dataLoaded % compute the projected boundaries
            if geometric
                filenameBounds = strcat('rawData/D_geom-',num2str(geometric),'_rm-',num2str(g{1}.meanR),'_S-',num2str(Smax),'-',num2str(resS),'_g-',num2str(gamm),'_t-',num2str(tmax),singleDecisionsSuffix,'_u-',utility,'.mat');
            else
                filenameBounds = strcat('rawData/D_geom-',num2str(geometric),'_rm-',num2str(g{1}.meanR),'_S-',num2str(Smax),'-',num2str(resS),'_c-',num2str(c),'_t-',num2str(tmax),singleDecisionsSuffix,'_u-',utility,'.mat');
            end
            if ~computeDecisionBoundaries && ~boundariesDataLoaded % load the decision threshold for all timesteps (matrix D)
                try
                    fprintf('loading boundaries...');
                    load(filenameBounds, 'D','rho_');
                    [S{1},S{2},S{3}] = ndgrid(Sscale, Sscale, Sscale); % define the space of possible rewards
                    boundariesDataLoaded = true;
                    fprintf('done.\n');
                catch
                    disp('Could not load the decision matrix. Recomputing the data.');
                end
            end
            if ~boundariesDataLoaded % compute the decision threshold for all timesteps
                fprintf('computing boundaries...');
                if singleDecisions
                    rho_ = 0; % we assume single decisions
                else
                    SscaleL  = linspace(-Smax, Smax, resSL);
                    [SLow{1},SLow{2},SLow{3}] = ndgrid(SscaleL, SscaleL, SscaleL);
                    iS0 = [findnearest(g{1}.meanR, SscaleL) findnearest(g{2}.meanR, SscaleL) findnearest(g{3}.meanR, SscaleL)]; % this line is not really needed
                    for iC = 3:-1:1;  RhLow{iC} = utilityFunc(SLow{iC});  end  % Expected reward for option iC
                    rho_ = fzero(@(rho) backwardInduction(rho,c,tNull,g,RhLow,SLow,t,dt,iS0), g{1}.meanR); % compute future rewards decisions
                    fprintf('rho for mean %d is %d...',g{1}.meanR, rho_);
                end
                iS0 = [findnearest(g{1}.meanR, Sscale) findnearest(g{2}.meanR, Sscale) findnearest(g{3}.meanR, Sscale)]; % this line is useless
                [V0, V, D, EVnext, rho, Ptrans, iStrans] = backwardInduction(rho_,c,tNull,g,Rh,S,t,dt,iS0);
                boundariesDataLoaded=true;
                fprintf('saving boundaries to file...');
                save(filenameBounds,'D','rho_', '-v7.3');
                fprintf('done.\n');
            end
            fprintf('computing projection...');
            dbIS = compute3dBoundaries(S, D(:,:,:,iT)); % computing the projected boundaries on 2d plane
            fprintf('saving projection to file...');
            save(filename,'dbIS','-v7.3');
            fprintf('done.\n');
        end
        % select subplot
        if magnitudePlot
        	subplot(2,length(timeSnaps),j); % select subplot
        else
            subplot(length(meanValues),length(timeSnaps),j); % select subplot
        end
        
        % convert to ternary plot coordinates
        for iD = 3:-1:1 
            if isempty(dbIS{iD}.vertices) == 0 
                dbIS{iD}.vertices2d = [ [ Smax/2 + (dbIS{iD}.vertices(:,2)-valscale) + (dbIS{iD}.vertices(:,3)-valscale)*0.5] [ Smax/(2*sqrt(3)) + (sqrt(3) * (dbIS{iD}.vertices(:,3)-valscale))/2] ]; % convert to ternary plot coordinates
                %dbIS{iD}.vertices2d = [ [ Smax + (dbIS{iD}.vertices(:,2)-valscale) + (dbIS{iD}.vertices(:,3)-valscale)*0.5] [ Smax/(sqrt(3)) + (sqrt(3) * (dbIS{iD}.vertices(:,3)-valscale))/2] ]; % convert to ternary plot coordinates
                %dbIS{iD}.vertices2d = (dbIS{iD}.vertices - valscale) * [1/sqrt(2) -1/sqrt(2) 0; -1/sqrt(3) -1/sqrt(3) 1/sqrt(3); 0 0 0]'; % (JARM 13th October '19) correct projection of decision thresholds when values scale
            end
        end
        
        % compute external triangle cropping at Smax/2
        zoom=Smax/3;
        attractor.vertices = [[zoom;-zoom;-zoom] + valscale, [-zoom;zoom;-zoom] + valscale, [-zoom;-zoom;zoom] + valscale]; 
        %attractor.vertices = [[Smax/2;-Smax/2;-Smax/2] + valscale, [-Smax/2;Smax/2;-Smax/2] + valscale, [-Smax/2;-Smax/2;Smax/2] + valscale]; 
        triang = [ [ zoom/(2) + (attractor.vertices(:,2)-valscale) + (attractor.vertices(:,3)-valscale)/2] [ zoom/(2*sqrt(3)) + (sqrt(3) * (attractor.vertices(:,3)-valscale))/2] ];
        vax = triang(1,1);  vay = triang(1,2); vbx = triang(2,1);  vby = triang(2,2); vcx = triang(3,1);  vcy = triang(3,2); % the ternary plot triangle vertices
        if ~magnitudePlot; patch([vax vbx vbx vax],[vay vay vcy vcy],'w','EdgeColor','none'); end % white background
        
        % plot the boundaries
        j=j+1;
        for iD = 1:3
            if isempty(dbIS{iD}) == 0 % (JARM 13th October '19) scaling value may lead to null intersection with decision boundaries
                if magnitudePlot
                    colour = 1-((valk/length(meanValues)) *[1 1 1]);
                else
                    colour = myCol(iD,1:3);
                end
                line([dbIS{iD}.vertices2d(dbIS{iD}.edges(:,1),1), dbIS{iD}.vertices2d(dbIS{iD}.edges(:,2),1)]',...
                    [dbIS{iD}.vertices2d(dbIS{iD}.edges(:,1),2), dbIS{iD}.vertices2d(dbIS{iD}.edges(:,2),2)]',...
                    'Color',colour,'LineWidth',2); hold on;
            end
        end
        %text(0,0,['t=' num2str(t(iT)) ' v=' num2str(meanValue)],'FontSize',10)
        
        % plot the external triangle
        patch([vax vcx vax],[vay vcy vcy],'w','EdgeColor','w','LineWidth',2); % clear out the line outside the triangle
        patch([vbx vcx vbx],[vby vcy vcy],'w','EdgeColor','w','LineWidth',2); % clear out the line outside the triangle
        line([vax vbx vcx vax], [vay vby vcy vay], 'Color',.5*[1 1 1],'LineWidth',1.5); 
        axis([vax vbx vay vcy ])
        %axis([-0.5 0.5 -0.5 0.5 ])
        set(gca,'visible','off') % remove axes
        daspect([1 1 1]); % aspect ratio 1
        
        % setting the time ticks
        if (valk==1)
            if magnitudePlot; x = 0.24*(timetk)+.03; txtsize=16; else; x = 0.22*(timetk)+.09;; txtsize=13; end
            
            timetk=timetk+1;
            annotation('ellipse', [0.055+x+0.06, 0.92, 0.01, 0.01], 'Color', 'k', 'FaceColor', 'k', 'LineWidth', 2, 'Units', 'normalized');
            annotation('textbox', [0.015+x+0.06, 0.81, 0.1, 0.1], 'String', num2str(t(iT)), 'EdgeColor', 'none', 'FontSize', txtsize, 'HorizontalAlignment', 'center');
        end
    end
end
end
ha=get(gcf,'children');
% Positioning the subplots and vertical ticks
if ~magnitudePlot
    for i = 1:length(meanValues)
        if magnitudePlot; break; end
        y = 0.12+0.25*(i-1);
        annotation('ellipse', [0.095, y+0.125, 0.01, 0.01], 'Color', 'k', 'FaceColor', 'k', 'LineWidth', 2, 'Units', 'normalized');
        annotation('textbox', [0.085, y+0.06, 0.1, 0.1], 'String', num2str(meanValues(length(meanValues)-i+1)), 'EdgeColor', 'none', 'FontSize', 13, 'HorizontalAlignment', 'center');
        for j = 1:length(timeSnaps)
            x = 0.22*(4-j)+.1;
            set(ha((i-1)*4+j),'position',[ x y .23 .23]); %set subplot positions
        end
    end
else
    for i = 1:2
        y = 0.02+0.405*(i-1);
        for j = 1:length(timeSnaps)
            x = 0.24*(4-j)-0.05;
            %fprintf("%f\n",x);
            set(ha((i-1)*4+j),'position',[ x y .4 .4]); %set subplot positions
        end
    end
end
%Arrows and labels
annotation('arrow', [0.11,0.95],[0.93,0.93],  'LineWidth', 3, 'Color', 'k');
if magnitudePlot; x=0.07; else; x=0.15; end
annotation('textbox', [x, 0.91, 0.1, 0.1], 'String', 'Time', 'EdgeColor', 'none', 'FontSize', 18, 'HorizontalAlignment', 'center');
if ~magnitudePlot
    stringlist=['V','a','l','u','e'];
    annotation('arrow', [0.1,0.1],[0.92,0.12],  'LineWidth', 3, 'Color', 'k');
else
    stringlist=['L','i','n','e','a','r'];
    stringlist2=['G','e','o','m','e','t','r','i','c'];
    for i = 1:length(stringlist2)
        annotation('textbox', [0, 0.43-(i*0.045), 0.05, 0.05], 'String', stringlist2(i), 'EdgeColor', 'none', 'FontSize', 18, 'HorizontalAlignment', 'center' );
    end
end
for i = 1:length(stringlist)
    annotation('textbox', [~magnitudePlot*0.05, 0.85-(i*0.045), 0.05, 0.05], 'String', stringlist(i), 'EdgeColor', 'none', 'FontSize', 18, 'HorizontalAlignment', 'center' );
end
%Save file
if magnitudePlot; sfx=''; else sfx='-mp'; end
outfilename = strcat('simFigs/vs-matrix',sfx,'-g',num2str(geometric),strjoin(strrep(suffix,'.',''),''),'.pdf');
set(gcf,'Color',[1 1 1]); set(gca,'Color',[.8 .8 .8]); set(gcf,'InvertHardCopy','off');
saveas(gcf,outfilename)
end

%% test one plot 
clf()

Smax=max(max(max(S{1})));
attractor.vertices = [[Smax;-Smax;-Smax] + valscale, [-Smax;Smax;-Smax] + valscale, [-Smax;-Smax;Smax] + valscale]; % (JARM 7th October '19 move triangle along diagonal as option values scale)
attractor.faces = [1 2 3; 1 2 3; 1 2 3];
for iD = 3:-1:1
    %[~, dbIS{iD}] = SurfaceIntersection(db{iD}, attractor);
    if isempty(dbIS{iD}.vertices) == 0 % (JARM 13th October '19) scaling value may lead to null intersection with decision boundaries
        %norma = @(a) a./norm(a); % define function to normalise vectors to length 1
        %prjMat=[ norma( cross([1 1 1], [1 2 4]) ); norma( cross( cross([1 1 1],[1 2 4]),[1,1,1]) )]; % compute the projection matrix to project the 3d coordinated onto the 2d plane ortogonal to the diagonal
        %dbIS{iD}.vertices2d = (prjMat * (dbIS{iD}.vertices(:,1:3)-valscale).').'; % project the 3d boundaries onto the 2d projection plane
        dbIS{iD}.vertices2d = [ [ Smax/2 + (dbIS{iD}.vertices(:,2)-valscale) + (dbIS{iD}.vertices(:,3)-valscale)*0.5] [ Smax/(2*sqrt(3)) + (sqrt(3) * (dbIS{iD}.vertices(:,3)-valscale))/2] ]; % convert to ternary plot coordinates
        %dbIS{iD}.vertices2d = (dbIS{iD}.vertices - valscale) * [1/sqrt(2) -1/sqrt(2) 0; -1/sqrt(3) -1/sqrt(3) 1/sqrt(3); 0 0 0]'; % (JARM 13th October '19) correct projection of decision thresholds when values scale
    end
end

%triang = [ [ Smax/2 + (attractor.vertices(:,2)-valscale) + (attractor.vertices(:,3)-valscale)*0.5] [ Smax/(2*sqrt(3)) + (sqrt(3) * (attractor.vertices(:,3)-valscale))/2] ];
attractor.vertices = [[Smax/2;-Smax/2;-Smax/2] + valscale, [-Smax/2;Smax/2;-Smax/2] + valscale, [-Smax/2;-Smax/2;Smax/2] + valscale]; 
triang = [ [ Smax/4 + (attractor.vertices(:,2)-valscale) + (attractor.vertices(:,3)-valscale)*0.5] [ Smax/(4*sqrt(3)) + (sqrt(3) * (attractor.vertices(:,3)-valscale))/2] ]; 
vax = triang(1,1);  vay = triang(1,2);
vbx = triang(2,1);  vby = triang(2,2);
vcx = triang(3,1);  vcy = triang(3,2);
patch([vax vbx vbx vax],[vay vay vcy vcy],'w','EdgeColor','none'); % white background

for iD = 1:3
    if isempty(dbIS{iD}) == 0 % (JARM 13th October '19) scaling value may lead to null intersection with decision boundaries
        line([dbIS{iD}.vertices2d(dbIS{iD}.edges(:,1),1), dbIS{iD}.vertices2d(dbIS{iD}.edges(:,2),1)]',...
            [dbIS{iD}.vertices2d(dbIS{iD}.edges(:,1),2), dbIS{iD}.vertices2d(dbIS{iD}.edges(:,2),2)]',...
            'Color',myCol(iD,1:3),'LineWidth',2); hold on;
    end
end

patch([vax vcx vax],[vay vcy vcy],'w','EdgeColor','none'); % clear out the line outside the triangle
patch([vbx vcx vbx],[vby vcy vcy],'w','EdgeColor','none'); % clear out the line outside the triangle
line([vax vbx vcx vax], [vay vby vcy vay], 'Color',.5*[1 1 1],'LineWidth',1.5);
axis([vax vbx vay vcy ])
daspect([1 1 1]); % aspect ratio 1
set(gca,'visible','off')

%% Sim code
numruns=1;
plotResults = true;

figure();
clf;
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [8 numruns*2]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 8 numruns*2]);

for run = 1:numruns
    r1sum=0;
    r2sum=0;
    r3sum=0;
    simTraj = [ ];
    for simT = 0:dtsim:tmax
        iT = findnearest(simT, t);
        r1m = ( (g{1}.meanR * g{1}.varX) + (r1sum * g{1}.varR) ) / (g{1}.varX + t(iT) * g{1}.varR );
        r2m = ( (g{2}.meanR * g{2}.varX) + (r2sum * g{2}.varR) ) / (g{2}.varX + t(iT) * g{2}.varR );
        r3m = ( (g{3}.meanR * g{3}.varX) + (r3sum * g{3}.varR) ) / (g{3}.varX + t(iT) * g{3}.varR );
        r1i = findnearest(r1m, Sscale, -1);
        r2i = findnearest(r2m, Sscale, -1);
        r3i = findnearest(r3m, Sscale, -1);
        dec=D(r1i,r2i,r3i,iT);
        if isempty(dec); dec=0; end
        simTraj = [simTraj; r1m r2m r3m dec ];
        %fprintf('For values (%d,%d,%d) at time %d the D is %d\n',r1m,r2m,r3m, t(iT), dec )
        if D(r1i,r2i,r3i,iT) ~= 4
            break
        else
            r1sum = r1sum + normrnd(g{1}.meanR*dtsim, sqrt(g{1}.varX*dtsim));
            r2sum = r2sum + normrnd(g{2}.meanR*dtsim, sqrt(g{2}.varX*dtsim));
            r3sum = r3sum + normrnd(g{3}.meanR*dtsim, sqrt(g{3}.varX*dtsim));
        end
    end
    if plotResults
        %subplot(ceil(numruns/2),2,run); imagesc(Sscale, Sscale, D(:,:,iT), [1 3]); axis square; axis xy; title(['D(0) \rho=' num2str(rho_,3)]); xlabel(Slabel{1}); ylabel(Slabel{2});
        valscale = (r1m + r2m + r3m + Smax)/3;
        %fprintf('valscale=%d',valscale);
        %valscale=0;
        dbIS = plotDecisionVolume(S, D(:,:,:,iT), [-Smax Smax] );
        hold on; plot3(simTraj(:,1),simTraj(:,2),simTraj(:,3),'k','linewidth',2);
        hold on; plot3(r1m,r2m,r3m,'ko','linewidth',2);
        %filename = strcat('simFigs/geometric-',num2str(geometric),'_r',num2str(run),'.pdf');
    end
end
%filename = strcat('simFigs/geometric-',num2str(geometric),'_r1-',num2str(g{1}.meanR),'_r2-',num2str(g{2}.meanR),'.pdf');
%savefig(filename)
%saveas(gcf,filename)

%% Giovanni's 2D plot - temporary test code
figure()
%patch([-sqrt(2) sqrt(2) 0 -sqrt(2)], [-1/sqrt(3) -1/sqrt(3) sqrt(3) -1/sqrt(3)],'w', 'EdgeColor',0.5*[1 1 1]);
%prjMat=[[1;1;0], [0;1;1], [1;0;1]];
%prjMat=[[0;0;1]-[0;1;0] [1;0;0]-[0;1;0]];
%prjMat=[ normalize([2 -3 1]); normalize([-1.5119 -0.3780 1.8898])];
%prjMat=[ normalize([2 -3 1]); normalize([1.5119 0.3780 -1.8898])];
norma = @(a) a./norm(a);
%prjMat=[ norma( cross([1 2 4], [1 1 1]) ); norma( cross( [1 1 1], cross([1 2 4], [1 1 1])) )];
prjMat=[ norma( cross([1 1 1], [1 2 4]) ); norma( cross( cross([1 1 1],[1 2 4]),[1,1,1]) )];
%prjMat=[[1 0 0]-[0 1 0]; [0 0 1]-[0 1 0]];
%prjMat= prjMat * (prjMat.' * prjMat)^(-1) * prjMat.';
%prjMat= prjMat * prjMat.';
simTrajP = prjMat * simTraj(:,1:3).';
%simTrajP = simTraj.';
%simTrajP = [ simTrajP(2,:) + simTrajP(3,:)*0.5 ; (sqrt(3) * simTrajP(3,:))/2 ]; % convert to ternary plot coordinates
for iD = 1:3
    %if isempty(dbIS{iD}) == 0 % (JARM 13th October '19) scaling value may lead to null intersection with decision boundaries
        %dbIS{iD}.vertices2d = [ [ Smax/2 + dbIS{iD}.vertices(:,2) + dbIS{iD}.vertices(:,3)*0.5] [ Smax/(2*sqrt(3)) + (sqrt(3) * dbIS{iD}.vertices(:,3))/2] ]; % convert to ternary plot coordinates
        %dbIS{iD}.vertices2d = (dbIS{iD}.vertices) * [1/sqrt(2) -1/sqrt(2) 0; -1/sqrt(3) -1/sqrt(3) 1/sqrt(3); 0 0 0]';
        %dbIS{iD}.vert = dbIS{iD}.vertices-valscale;
        %dbIS{iD}.vertices2d = [ [ Smax/2 + dbIS{iD}.vert(:,2) + dbIS{iD}.vert(:,3)*0.5] [ Smax/(2*sqrt(3)) + (sqrt(3) * dbIS{iD}.vert(:,3))/2] ]; % convert to ternary plot coordinates
        %dbIS{iD}.vertices2d = [ [ Smax/2 + (dbIS{iD}.vertices(:,2)-valscale) + (dbIS{iD}.vertices(:,3)-valscale)*0.5] [ Smax/(2*sqrt(3)) + (sqrt(3) * (dbIS{iD}.vertices(:,3)-valscale))/2] ]; % convert to ternary plot coordinates
        dbIS{iD}.vertices2d = (prjMat * (dbIS{iD}.vertices(:,1:3)-valscale).').';
        %dbIS{iD}.vertices2d = dbIS{iD}.vertices;
        %dbIS{iD}.vertices2d = (dbIS{iD}.vertices - valscale) * [1/sqrt(2) -1/sqrt(2) 0; -1/sqrt(3) -1/sqrt(3) 1/sqrt(3); 0 0 0]';
        line([dbIS{iD}.vertices2d(dbIS{iD}.edges(:,1),1), dbIS{iD}.vertices2d(dbIS{iD}.edges(:,2),1)]',...
             [dbIS{iD}.vertices2d(dbIS{iD}.edges(:,1),2), dbIS{iD}.vertices2d(dbIS{iD}.edges(:,2),2)]',...
             'Color',myCol(iD,1:3),'LineWidth',1.5); hold on;
         %[dbIS{iD}.vertices2d(dbIS{iD}.edges(:,1),3), dbIS{iD}.vertices2d(dbIS{iD}.edges(:,2),3)]', ...
    %end
end
%clf;
hold on; plot(simTrajP(1,:),simTrajP(2,:),'k');
hold on; plot(simTrajP(1,length(simTraj)),simTrajP(2,length(simTraj)),'or');
hold on; plot(0,0,'ok');
daspect([1 1 1]); % aspect ratio 1
title(['time=' num2str(t(iT)) ' -- \Sigma_i(r_i)=' num2str(sum(simTraj(iT,1:3))) ]);

%% Plot intermediate steps
stepsize = 10;
timesteps = stepsize:stepsize:length(simTraj);
if timesteps(length(timesteps))~=length(simTraj) timesteps = [timesteps length(simTraj)]; end
figure();
clf;
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [8 length(timesteps)*2]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 8 length(timesteps)*2]);

for iT = timesteps
    valscale = (sum(simTraj(iT,1:3)) + Smax)/3;
    dbIS = compute3dBoundaries(S, D(:,:,:,iT));
    
    subplot(ceil(length(timesteps)/2),2,findnearest(iT,timesteps));
    plotTrajOnProjection(dbIS, simTraj(1:iT,:), myCol);
    title(['time=' num2str(t(iT)) ' -- \Sigma_i(r_i)=' num2str(sum(simTraj(iT,1:3))) ]);
end
filename = strcat('simFigs/fullSim-geometric-',num2str(geometric),'_r1-',num2str(g{1}.meanR),'_r2-',num2str(g{2}.meanR),'_r3-',num2str(g{3}.meanR),'.pdf');
%savefig(filename)
saveas(gcf,filename)

%% Visualise collapsing boundaries
computeDecisionBoundaries=false; % if true the code will compute the decision boundaries, if false will try to load the decision boundaries from the directory rawData (if it fails, it will recompute the data)
singleDecisions=true; % if true we model single decision (i.e. expected future rewards=0); if false we compute the expected future reward (rho_)
cuttingPlanes=0:1:3; % cutting plane, the value corresponds to the sum of the expected rewards (r1+r2+r3)
listOfTimes=1:100:601; % the snapshots of when to show the boundaries
priorMean=1.5;
if singleDecisions
    singleDecisionsSuffix='-singleDec';
else
    singleDecisionsSuffix='-multiDec';
end

boundariesDataLoaded = false;

% prepare the figure
figure(); clf;
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [2+2*length(listOfTimes) 2+length(cuttingPlanes)*2]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 2+2*length(listOfTimes) 2+length(cuttingPlanes)*2]);
j=1; % counter for subplots
for cuttingPlane = cuttingPlanes
    valscale = cuttingPlane + (Smax/3); % moving to the correct 2d projection plane
    for snapshot = listOfTimes
        if geometric
            filename = strcat('rawData/plane_geom-',num2str(geometric),'_rm-',num2str(priorMean),'_S-',num2str(Smax),'-',num2str(resS),'_g-',num2str(gamm),'_t-',num2str(tmax),singleDecisionsSuffix,'_plane-',num2str(cuttingPlane),'_snap-',num2str(snapshot),'.mat');
        else
            filename = strcat('rawData/plane_geom-',num2str(geometric),'_rm-',num2str(priorMean),'_S-',num2str(Smax),'-',num2str(resS),'_c-',num2str(c),'_t-',num2str(tmax),singleDecisionsSuffix,'_plane-',num2str(cuttingPlane),'_snap-',num2str(snapshot),'.mat');
        end
        % try to load the file
        dataLoaded = false;
        if ~computeDecisionBoundaries % load the projected boundaries
            try
                fprintf('loading projection...');
                load(filename, 'dbIS');
                dataLoaded = true;
                fprintf('done.\n');
            catch
                disp('Could not load the projected boundaries. Recomputing the data.');
            end
        end
        if ~dataLoaded % compute the projected boundaries
            if geometric
                filenameBounds = strcat('rawData/D_geom-',num2str(geometric),'_rm-',num2str(priorMean),'_S-',num2str(Smax),'-',num2str(resS),'_g-',num2str(gamm),'_t-',num2str(tmax),singleDecisionsSuffix,'.mat');
            else
                filenameBounds = strcat('rawData/D_geom-',num2str(geometric),'_rm-',num2str(priorMean),'_S-',num2str(Smax),'-',num2str(resS),'_c-',num2str(c),'_t-',num2str(tmax),singleDecisionsSuffix,'.mat');
            end
            if ~computeDecisionBoundaries && ~boundariesDataLoaded % load the decision threshold for all timesteps (matrix D)
                try
                    fprintf('loading boundaries...');
                    load(filenameBounds, 'D','rho_');
                    [S{1},S{2},S{3}] = ndgrid(Sscale, Sscale, Sscale); % define the space of possible rewards
                    boundariesDataLoaded = true;
                    fprintf('done.\n');
                catch
                    disp('Could not load the decision matrix. Recomputing the data.');
                end
            end
            if ~boundariesDataLoaded % compute the decision threshold for all timesteps
                fprintf('computing boundaries...');
                g{1}.meanR = priorMean; g{2}.meanR = priorMean; g{3}.meanR = priorMean; % fix prior of mean to a fixed value
                if singleDecisions
                    rho_ = 0; % we assume single decisions
                else
                    SscaleL  = linspace(-Smax, Smax, resSL);
                    [SLow{1},SLow{2},SLow{3}] = ndgrid(SscaleL, SscaleL, SscaleL);
                    iS0 = [findnearest(g{1}.meanR, SscaleL) findnearest(g{2}.meanR, SscaleL) findnearest(g{3}.meanR, SscaleL)]; % this line is not really needed
                    for iC = 3:-1:1;  RhLow{iC} = utilityFunc(SLow{iC});  end  % Expected reward for option iC
                    rho_ = fzero(@(rho) backwardInduction(rho,c,tNull,g,RhLow,SLow,t,dt,iS0), g{1}.meanR); % compute future rewards decisions
                    fprintf('rho for mean %d is %d...',g{1}.meanR, rho_);
                end
                
                Sscale = linspace(-Smax, Smax, resS); % define the range of possible rewards
                [S{1},S{2},S{3}] = ndgrid(Sscale, Sscale, Sscale); % define the space of possible rewards
                Rh{1} = utilityFunc(S{1}); % Expected reward for option 1
                Rh{2} = utilityFunc(S{2}); % Expected reward for option 2
                Rh{3} = utilityFunc(S{3}); % Expected reward for option 3
                iS0 = [findnearest(g{1}.meanR, Sscale) findnearest(g{2}.meanR, Sscale) findnearest(g{3}.meanR, Sscale)]; % this line is not really needed
                [V0, V, D, EVnext, rho, Ptrans, iStrans] = backwardInduction(rho_,c,tNull,g,Rh,S,t,dt,iS0);
                boundariesDataLoaded=true;
                fprintf('saving boundaries to file...');
                save(filenameBounds,'D','rho_', '-v7.3');
                fprintf('done.\n');
            end
            fprintf('computing projection...');
            dbIS = compute3dBoundaries(S, D(:,:,:,snapshot)); % computing the projected boundaries on 2d plane
            fprintf('saving projection to file...');
            save(filename,'dbIS','-v7.3');
            fprintf('done.\n');
        end
        
        subplot(length(cuttingPlanes),length(listOfTimes),j); % select subplot
        j=j+1;
        for iD = 1:3
            if isempty(dbIS{iD}) == 0 % scaling value may lead to null intersection with decision boundaries
                line([dbIS{iD}.vertices2d(dbIS{iD}.edges(:,1),1), dbIS{iD}.vertices2d(dbIS{iD}.edges(:,2),1)]',...
                    [dbIS{iD}.vertices2d(dbIS{iD}.edges(:,1),2), dbIS{iD}.vertices2d(dbIS{iD}.edges(:,2),2)]',...
                    'Color',myCol(iD,1:3),'LineWidth',1.5); hold on;
            end
        end
        xlim([-Smax-1 Smax+1]); ylim([-Smax-1 Smax+1]); daspect([1 1 1]); % aspect ratio 1
        title(['time=' num2str(t(snapshot)) ' -- \Sigma_i(r_i)=' num2str(cuttingPlane) ]);
    end
end
fprintf('saving image to pdf...');
filename = strcat('simFigs/collapsing_geom-',num2str(geometric),'_pm-',num2str(priorMean),'.pdf');
saveas(gcf,filename)
fprintf('done.\n');

%% Plot Trajectory on 2d projection
function plotTrajOnProjection(dbIS, simTraj, myCol)
for iD = 1:3
    if isempty(dbIS{iD}) == 0 % (JARM 13th October '19) scaling value may lead to null intersection with decision boundaries
        line([dbIS{iD}.vertices2d(dbIS{iD}.edges(:,1),1), dbIS{iD}.vertices2d(dbIS{iD}.edges(:,2),1)]',...
             [dbIS{iD}.vertices2d(dbIS{iD}.edges(:,1),2), dbIS{iD}.vertices2d(dbIS{iD}.edges(:,2),2)]',...
             'Color',myCol(iD,1:3),'LineWidth',1.5); hold on;
    end
end
norma = @(a) a./norm(a); % define function to normalise vectors to length 1
%prjMat=[ norma( cross([1 2 4], [1 1 1]) ); norma( cross( [1 1 1], cross([1 2 4], [1 1 1])) )];
prjMat=[ norma( cross([1 1 1], [1 2 4]) ); norma( cross( cross([1 1 1],[1 2 4]),[1,1,1]) )]; % compute the projection matrix to project the 3d trajectory onto the 2d plane ortogonal to the diagonal
simTrajP = prjMat * simTraj(:,1:3).'; % convert the trajectoy's 3d coordinate into 2d (projected) coordiantes
hold on; plot(simTrajP(1,:),simTrajP(2,:),'k'); % plot trajectory
hold on; plot(simTrajP(1,length(simTraj)),simTrajP(2,length(simTraj)),'or') % plot dot at final position
hold on; plot(simTrajP(1,1),simTrajP(1,1),'ok') % plot dot at initial position
return

%% Compute boundaries
function [dbIS] = compute3dBoundaries(S, D)
global geometric epsil valscale
for iD = 3:-1:1
    switch iD
        case 1
            idx = D==1 | D==12 | D==13 | D==123;
        case 2
            idx = D==2 | D==12 | D==23 | D==123;
        case 3
            idx = D==3 | D==23 | D==13 | D==123;
    end
      db{iD}.vertices = [vector(S{1}(idx)), vector(S{2}(idx)), vector(S{3}(idx))];
      if geometric
        db{iD}.faces = convhull(vector(S{1}(idx)) + epsil * randn(size(vector(S{1}(idx)))), vector(S{2}(idx)) + epsil * randn(size(vector(S{1}(idx)))), vector(S{3}(idx)) + epsil * randn(size(vector(S{1}(idx)))));
      else
        db{iD}.faces = convhull(vector(S{1}(idx)), vector(S{2}(idx)), vector(S{3}(idx)));
      end
end
Smax=max(max(max(S{1})));
attractor.vertices = [[Smax;-Smax;-Smax] + valscale, [-Smax;Smax;-Smax] + valscale, [-Smax;-Smax;Smax] + valscale]; % (JARM 7th October '19 move triangle along diagonal as option values scale)
attractor.faces = [1 2 3; 1 2 3; 1 2 3];
for iD = 3:-1:1
    [~, dbIS{iD}] = SurfaceIntersection(db{iD}, attractor);
    if isempty(dbIS{iD}.vertices) == 0 % (JARM 13th October '19) scaling value may lead to null intersection with decision boundaries
        norma = @(a) a./norm(a); % define function to normalise vectors to length 1
        prjMat=[ norma( cross([1 1 1], [1 2 4]) ); norma( cross( cross([1 1 1],[1 2 4]),[1,1,1]) )]; % compute the projection matrix to project the 3d coordinated onto the 2d plane ortogonal to the diagonal
        dbIS{iD}.vertices2d = (prjMat * (dbIS{iD}.vertices(:,1:3)-valscale).').'; % project the 3d boundaries onto the 2d projection plane
        %dbIS{iD}.vertices2d = [ [ Smax/2 + (dbIS{iD}.vertices(:,2)-valscale) + (dbIS{iD}.vertices(:,3)-valscale)*0.5] [ Smax/(2*sqrt(3)) + (sqrt(3) * (dbIS{iD}.vertices(:,3)-valscale))/2] ]; % convert to ternary plot coordinates
        %dbIS{iD}.vertices2d = (dbIS{iD}.vertices - valscale) * [1/sqrt(2) -1/sqrt(2) 0; -1/sqrt(3) -1/sqrt(3) 1/sqrt(3); 0 0 0]'; % (JARM 13th October '19) correct projection of decision thresholds when values scale
    end
end
return

%% Giovanni's test plots
figure();
cuttingplane=0;
rect = [-1 1 -1 1 -2.2 1.5];
iT = 2;
iS2 = findnearest(cuttingplane, Sscale, -1);
iS3 = 1;
[r1Max,r2Max,vMax] = plotSurf(Sscale, V(:,:,iS3,iT), iS2, [0 0 0], Slabel); axis(rect); title('V(0)');
if geometric
  [r1Acc,r2Acc,vAcc] = plotSurf(Sscale, EVnext(:,:,iS3,iT), iS2, [1 0 0], Slabel); axis(rect); title('<V(\deltat)|R^{hat}(0)> - (\rho+c)\deltat'); % (JARM 23rd August '19) remove cost of time since incorporated into discounted reward rate
else
  [r1Acc,r2Acc,vAcc] = plotSurf(Sscale, EVnext(:,:,iS3,iT)-(rho+c)*dt, iS2, [1 0 0], Slabel); axis(rect); title('<V(\deltat)|R^{hat}(0)> - (\rho+c)\deltat');
end
[r1Dec,r2Dec,vDec] = plotSurf(Sscale, RhMax(:,:,iS3)-rho*tNull, iS2, [0 0 1], Slabel); axis(rect); title('max(R_1^{hat},R_2^{hat}) - \rho t_{Null}');
% surfl(Sscale, Sscale, ones(length(Sscale),length(Sscale))*cuttingplane); hold on;
figure()
plot([-1,1],[1-c*(dt*iT)-rho*tNull,1-c*(dt*iT)-rho*tNull],'k',(r1Max-r2Max)/2, vMax, 'k:', (r1Acc-r2Acc)/2, vAcc, 'r', (r1Dec-r2Dec)/2, vDec, 'b'); xlabel(['(' Slabel{1} '-' Slabel{2} ')/2']); xlim(rect(1:2));
figure()
xvals=-4:0.2:4
yvals=abs(xvals)-rho*tNull
plot(r1Max, vMax, 'k:', r1Acc, vAcc, 'r', r1Dec, vDec, 'b'); xlabel(['(' Slabel{1} '-' Slabel{2} ')/2']); xlim([-Smax Smax]);
%plot(xvals,yvals,'g')

%% - Show -
figure(4565); clf; colormap bone;
iS2 = findnearest(.5, Sscale, -1);
iS3 = 1;
iTmax = length(t);
rect = [-1 1 -1 1 -2.3 .5];
myCol = [1 0 0; 0 1 0; 0 0 1];
%% t=0:
iT = 1;
subplotXY(5,4,2,1); [r1Max,r2Max,vMax] = plotSurf(Sscale, V(:,:,iS3,iT)                , iS2, [0 0 0], Slabel); axis(rect); title('V(0)');
%                     plot3(g{1}.meanR, g{2}.meanR, V0, 'g.', 'MarkerSize',15);
if geometric
  subplotXY(5,4,3,1); [r1Acc,r2Acc,vAcc] = plotSurf(Sscale, EVnext(:,:,iS3,iT), iS2, [1 0 0], Slabel); axis(rect); title('<V(\deltat)|R^{hat}(0)> - (\rho+c)\deltat'); % (JARM 23rd August '19) remove cost of time since incorporated into discounted reward rate
else
  subplotXY(5,4,3,1); [r1Acc,r2Acc,vAcc] = plotSurf(Sscale, EVnext(:,:,iS3,iT)-(rho+c)*dt, iS2, [1 0 0], Slabel); axis(rect); title('<V(\deltat)|R^{hat}(0)> - (\rho+c)\deltat');
end
subplotXY(5,4,4,1); [r1Dec,r2Dec,vDec] = plotSurf(Sscale, RhMax(:,:,iS3)-rho*tNull     , iS2, [0 0 1], Slabel); axis(rect); title('max(R_1^{hat},R_2^{hat}) - \rho t_{Null}');
subplotXY(5,4,5,1); hold on;
    plot((r1Max-r2Max)/2, vMax, 'k:', (r1Acc-r2Acc)/2, vAcc, 'r', (r1Dec-r2Dec)/2, vDec, 'b');
    xlabel(['(' Slabel{1} '-' Slabel{2} ')/2']); xlim(rect(1:2)); %ylim(rect(5:6));
subplotXY(5,4,1,1); imagesc(Sscale, Sscale, D(:,:,  1), [1 4]); axis square; axis xy; title(['D(0) \rho=' num2str(rho_,3)]); xlabel(Slabel{1}); ylabel(Slabel{2}); hold on; axis(rect(1:4));
                    plot(r1Max, r2Max, 'r-');
%                     plot(g{1}.meanR, g{2}.meanR, 'g.');
%% t=0 (superimposed & difference):
if geometric
  subplotXY(5,4,3,2); plotSurf(Sscale, EVnext(:,:,iS3,iT)                                        , iS2, [1 0 0], Slabel); hold on; % (JARM 23rd August '19) remove cost of time since incorporated into discounted reward rate    
else
  subplotXY(5,4,3,2); plotSurf(Sscale, EVnext(:,:,iS3,iT)-(rho+c)*dt                             , iS2, [1 0 0], Slabel); hold on;
end
                    plotSurf(Sscale, RhMax(:,:,iS3)-rho*tNull                                  , iS2, [0 0 1], Slabel); axis(rect);
if geometric
  subplotXY(5,4,4,2); plotSurf(Sscale, RhMax(:,:,iS3)-rho*tNull - EVnext(:,:,iS3,iT), iS2, [0 1 0], Slabel); xlim(rect(1:2)); ylim(rect(1:2)); % (JARM 23rd August '19) remove cost of time since incorporated into discounted reward rate     
else 
  subplotXY(5,4,4,2); plotSurf(Sscale, RhMax(:,:,iS3)-rho*tNull - (EVnext(:,:,iS3,iT)-(rho+c)*dt), iS2, [0 1 0], Slabel); xlim(rect(1:2)); ylim(rect(1:2));
end
%if geometric == false
subplotXY(5,4,5,2); plotDecisionVolume(S, D(:,:,:,iT), rect(1:2)); % (JARM 23rd August '19) error computing convex hull under geometric discounting
%end
%% t=dt:
subplotXY(5,4,1,2); imagesc(Sscale, Sscale, D(:,:,iS3,iT+1), [1 4]); axis square; axis xy; title('D(\deltat)'); xlabel(Slabel{1}); ylabel(Slabel{2}); hold on; axis(rect(1:4));
subplotXY(5,4,2,2); plotSurf(Sscale, V(:,:,iS3,iT+1), iS2, [0 0 0], Slabel); axis(rect); title('V(\deltat)');
%% t=T-dt:
% subplotXY(5,4,3,2); surfl(Sscale(iStrans{1}{2}), Sscale(iStrans{1}{2}), Ptrans{1}); title('P(R^{hat}(\deltat)|R^{hat}(0))'); shading interp; hold on; axis([rect 0 Inf]); axis off;
subplotXY(5,4,1,3); imagesc(Sscale, Sscale, D(:,:,iS3,iTmax-1), [1 4]); axis square; axis xy; title('D(T-\deltat)'); hold on; rectangle('Position',[rect(1) rect(3) rect(2)-rect(1) rect(4)-rect(3)]); axis(rect);
subplotXY(5,4,2,3); [r1Max,r2Max,vMax] = plotSurf(Sscale, V(:,:,iS3,iTmax-1)                , iS2, [0 0 0], Slabel); axis(rect); title('V(T-\deltat)')
if geometric
  subplotXY(5,4,3,3); [r1Acc,r2Acc,vAcc] = plotSurf(Sscale, EVnext(:,:,iS3,iTmax-1), iS2, [1 0 0], Slabel); axis(rect); title('<V(T)|R^{hat}(T-\deltat)>'); % (JARM 23rd August '19) remove cost of time since incorporated into discounted reward rate  
else
  subplotXY(5,4,3,3); [r1Acc,r2Acc,vAcc] = plotSurf(Sscale, EVnext(:,:,iS3,iTmax-1)-(rho+c)*dt, iS2, [1 0 0], Slabel); axis(rect); title('<V(T)|R^{hat}(T-\deltat)> - (\rho+c) \deltat');
end
subplotXY(5,4,4,3); [r1Dec,r2Dec,vDec] = plotSurf(Sscale, RhMax(:,:,iS3)-rho*tNull          , iS2, [0 0 1], Slabel); axis(rect); title('max(R_1^{Hat},R_2^{Hat}) - \rho t_{Null}');
subplotXY(5,4,5,3); hold on;
    plot((r1Max-r2Max)/2, vMax, 'k:', (r1Acc-r2Acc)/2, vAcc, 'r', (r1Dec-r2Dec)/2, vDec, 'b');
    xlabel(['(' Slabel{1} '-' Slabel{2} ')/2']); xlim(rect(1:2)); %ylim(rect(5:6));
%if geometric == false
  subplotXY(5,4,5,4); plotDecisionVolume(S, D(:,:,:,iTmax-1), rect(1:2)); % (JARM 23rd August '19) error computing convex hull under geometric discounting
%end
%% t=T:
subplotXY(5,4,1,4); imagesc(Sscale, Sscale, D(:,:,iS3,iTmax), [1 4]); axis square; axis xy; title('D(T)'); hold on; axis(rect(1:4));
subplotXY(5,4,2,4); plotSurf(Sscale, V(:,:,iS3,iTmax), iS2, [0 0 0], Slabel); title('V(T) = max(R_1^{hat},R_2^{hat}) - \rho t_{Null}'); axis(rect);
% subplotXY(5,4,3,4); surfl(Sscale(iStrans{iTmax-1}{2}), Sscale(iStrans{iTmax-1}{1}), Ptrans{iTmax-1}); title('P(R^{hat}(T)|R^{hat}(T-\deltat))'); shading interp; hold on; axis([rect 0 Inf]); axis off;


%% Decision boundaries superimposed:
figure(4566); clf;
iT = [1 3 11 41];
for iiT = 1:length(iT)
    subplot(2,1,1);
        plotDecisionVolume(S, D(:,:,:,iT(iiT)), rect(1:2), [myCol, [0.3; 0.05; 0.05]]); hold on;
    subplotXY(4,length(iT),3,iiT);
        dbIS = plotDecisionVolume(S, D(:,:,:,iT(iiT)), rect(1:2), [myCol, [0.5;0.5;0.5]]); hold on;
        title(['Time=' num2str(t(iT(iiT)))]);
    subplotXY(4,length(iT),4,iiT);
        patch([-sqrt(2) sqrt(2) 0 -sqrt(2)], [-1/sqrt(3) -1/sqrt(3) sqrt(3) -1/sqrt(3)],'w', 'EdgeColor',0.5*[1 1 1]);
        for iD = 1:3
            if isempty(dbIS{iD}) == 0 % (JARM 13th October '19) scaling value may lead to null intersection with decision boundaries
                line([dbIS{iD}.vertices2d(dbIS{iD}.edges(:,1),1), dbIS{iD}.vertices2d(dbIS{iD}.edges(:,2),1)]',...
                     [dbIS{iD}.vertices2d(dbIS{iD}.edges(:,1),2), dbIS{iD}.vertices2d(dbIS{iD}.edges(:,2),2)]',...
                     [dbIS{iD}.vertices2d(dbIS{iD}.edges(:,1),3), dbIS{iD}.vertices2d(dbIS{iD}.edges(:,2),3)]', ...
                     'Color',myCol(iD,1:3),'LineWidth',1.5); hold on;
            end
        end
        axis([-2 2 -2 2 -1 1]); view([0 90]); axis off;
end
toc;


function [V0, V, D, EVnext, rho, Ptrans, iStrans] = backwardInduction(rho_,c,tNull,g,Rh,S,t,dt,iS0)
global gamm geometric;
rho = rho_;                                                                        % Reward rate estimate
[V(:,:,:,length(t)), D(:,:,:,length(t))] = max_({Rh{1}-rho*tNull, Rh{2}-rho*tNull, Rh{3}-rho*tNull});       % Max V~ at time tmax
if geometric; gammaStep = gamm^dt; end
for iT = length(t)-1:-1:1
    [EVnext(:,:,:,iT), Ptrans{iT}, iStrans{iT}] = E(V(:,:,:,iT+1),S,t(iT),dt,g);                            % <V~(t+1)|S(t)> for waiting
    if geometric
      [V(:,:,:,iT), D(:,:,:,iT)] = max_({Rh{1}-rho*tNull, Rh{2}-rho*tNull, Rh{3}-rho*tNull, EVnext(:,:,:,iT)*gammaStep});       % (JARM 23rd August'19) [geometrically-discounted value (V~), decision] at time t    
    else
      [V(:,:,:,iT), D(:,:,:,iT)] = max_({Rh{1}-rho*tNull, Rh{2}-rho*tNull, Rh{3}-rho*tNull, EVnext(:,:,:,iT)-(rho+c)*dt});       % [Average-adjusted value (V~), decision] at time t
    end
%     fprintf('%d/%d\t',iT,length(t)-1); toc;
end
V0 = mean(vector(V(iS0(1),iS0(2),1)));
D(D==0) = 4;
fprintf('rho = %d\tV0 = %d\t', rho_, V0); toc;

function [EV, Ptrans, iStrans] = E(V,S,t,dt,g)
aSscale = abs(S{1}(:,1,1));
CR = [g{1}.varR 0 0; 0 g{2}.varR 0; 0 0 g{3}.varR];
CX = [g{1}.varX 0 0; 0 g{2}.varX 0; 0 0 g{3}.varX];
for k = 1:3
    g{k}.varRh = g{k}.varR * g{k}.varX / (t * g{k}.varR + g{k}.varX);
    v{k} = varTrans(g{k}.varRh, g{k}.varR, g{k}.varX, t, dt);
    iStrans{k} = find(aSscale<3*sqrt(v{k}));
end
Ptrans = normal3({S{1}(iStrans{1},iStrans{2},iStrans{3}), S{2}(iStrans{1},iStrans{2},iStrans{3}), S{3}(iStrans{1},iStrans{2},iStrans{3})}, [0 0 0], [v{1} 0 0; 0 v{2} 0; 0 0 v{3}]);
mgn = ceil(size(Ptrans)/2);
% V = extrap(V,mgn,[5 5 5]);
EV = convn(V,Ptrans,'same') ./ convn(ones(size(V)),Ptrans,'same');
% EV = EV(mgn(1)+1:end-mgn(1), mgn(2)+1:end-mgn(2), mgn(3)+1:end-mgn(3));

function v = varTrans(varRh, varR, varX, t, dt)
% v = (varR * (varX + varRh)) / ((1 + t/dt) * varR + varX / dt);
v = (varR ./ (varR*(t+dt) + varX)).^2 .* (varX + varRh * dt) * dt;

function prob = normal3(x, m, C)
d1 = x{1} - m(1);
d2 = x{2} - m(2);
d3 = x{3} - m(3);
H = -1/2*(C\eye(3)); prob = exp(d1.*d1*H(1,1) + d1.*d2*H(1,2) + d1.*d3*H(1,3) + ...
                                d2.*d1*H(2,1) + d2.*d2*H(2,2) + d2.*d3*H(2,3) + ...
                                d3.*d1*H(3,1) + d3.*d2*H(3,2) + d3.*d3*H(3,3));
% prob = exp(-(d1.^2/C(1,1)/2 + d2.^2/C(2,2))/2);
prob = prob ./ sum(prob(:));

function [V, D] = max_(x)
x_ = zeros(size(x{1},1), size(x{1},2), size(x{1},3), length(x));
for k = 1:length(x)
    x_(:,:,:,k) = x{k};
end
[V, D] = max(x_,[],4);
D(D==1 & x{1}==x{2} & x{2}==x{3}) = 123;
D(D==1 & x{1}==x{2}) = 12;
D(D==2 & x{2}==x{3}) = 23;
D(D==1 & x{3}==x{1}) = 13;

function [x_,y_,v_] = plotSurf(Sscale, Val, iS, col, Slabel)
[x,y] = meshgrid(1:length(Sscale), 1:length(Sscale));
x_ = Sscale(x(x+y==iS+round(length(Sscale)/2)));
y_ = Sscale(y(x+y==iS+round(length(Sscale)/2)));
v_ = Val(x+y==iS+round(length(Sscale)/2));
h = surfl(Sscale, Sscale, Val); hold on; %camproj perspective;
set(h,'FaceColor',sat(.5,col), 'EdgeColor','none'); camlight left; lighting phong; alpha(0.7);
if ischar(col);  plot3(x_, y_, v_,         col); hold on;
else             plot3(x_, y_, v_, 'Color',col); hold on;  end
xlabel(Slabel{1}); ylabel(Slabel{2}); %zlim([-50 50]);
% h = get(gca,'XLabel'); set(h,'FontSize',8, 'Position',get(h,'Position')+[0 .2 0]);
% h = get(gca,'YLabel'); set(h,'FontSize',8, 'Position',get(h,'Position')+[1 .2 0]);

function [dbIS] = plotDecisionVolume(S, D, minmax, myCol)
global geometric epsil valscale
if nargin < 4;  myCol = [1 0 0 0.5; 0 1 0 0.5; 0 0 1 0.5];  end
shiftMin = 0.01 * [1 0 0; 0 1 0; 0 0 1];
for iD = 3:-1:1
    switch iD
        case 1
            idx = D==1 | D==12 | D==13 | D==123;
        case 2
            idx = D==2 | D==12 | D==23 | D==123;
        case 3
            idx = D==3 | D==23 | D==13 | D==123;
    end
%      plot3(vector(S{1}(idx)), vector(S{2}(idx)), vector(S{3}(idx)), '.', 'Color', myCol(iD,:));
      db{iD}.vertices = [vector(S{1}(idx)), vector(S{2}(idx)), vector(S{3}(idx))];
      if geometric
%        size(vector(S{1}(idx)))
%        size(vector(S{2}(idx)))
%        size(vector(S{3}(idx)))
%        geometric
        db{iD}.faces = convhull(vector(S{1}(idx)) + epsil * randn(size(vector(S{1}(idx)))), vector(S{2}(idx)) + epsil * randn(size(vector(S{1}(idx)))), vector(S{3}(idx)) + epsil * randn(size(vector(S{1}(idx)))));
      else
        db{iD}.faces = convhull(vector(S{1}(idx)), vector(S{2}(idx)), vector(S{3}(idx)));
      end
      trisurf(db{iD}.faces, db{iD}.vertices(:,1)+shiftMin(iD,1), db{iD}.vertices(:,2)+shiftMin(iD,2), db{iD}.vertices(:,3)+shiftMin(iD,3), 'FaceColor',myCol(iD,1:3),'FaceAlpha',myCol(iD,4),'EdgeColor','none'); hold on;
end
%attractor.vertices = [[1;-1;-1] + valscale, [-1;1;-1] + valscale, [-1;-1;1] + valscale]; % (JARM 7th October '19 move triangle along diagonal as option values scale)
Smax=max(max(max(S{1})));
attractor.vertices = [[Smax;-Smax;-Smax] + valscale, [-Smax;Smax;-Smax] + valscale, [-Smax;-Smax;Smax] + valscale]; % (JARM 7th October '19 move triangle along diagonal as option values scale)
attractor.faces = [1 2 3; 1 2 3; 1 2 3];
trisurf(attractor.faces, attractor.vertices(:,1), attractor.vertices(:,2), attractor.vertices(:,3), 'FaceColor',[0 0 0],'FaceAlpha',0.1,'EdgeColor','none'); hold on;
for iD = 3:-1:1
    [~, dbIS{iD}] = SurfaceIntersection(db{iD}, attractor);
    if isempty(dbIS{iD}.vertices) == 0 % (JARM 13th October '19) scaling value may lead to null intersection with decision boundaries
        %dbIS{iD}.vertices2d = (dbIS{iD}.vertices - valscale) * [1/sqrt(2) -1/sqrt(2) 0; -1/sqrt(3) -1/sqrt(3) 1/sqrt(3); 0 0 0]'; % (JARM 13th October '19) correct projection of decision thresholds when values scale
        %dbIS{iD}.vertices2d = [ [ Smax/2 + (dbIS{iD}.vertices(:,2)-valscale) + (dbIS{iD}.vertices(:,3)-valscale)*0.5] [ Smax/(2*sqrt(3)) + (sqrt(3) * (dbIS{iD}.vertices(:,3)-valscale))/2] ]; % convert to ternary plot coordinates
        norma = @(a) a./norm(a); % define function to normalise vectors to length 1
        prjMat=[ norma( cross([1 1 1], [1 2 4]) ); norma( cross( cross([1 1 1],[1 2 4]),[1,1,1]) )]; % compute the projection matrix to project the 3d coordinated onto the 2d plane ortogonal to the diagonal
        dbIS{iD}.vertices2d = (prjMat * (dbIS{iD}.vertices(:,1:3)-valscale).').'; % project the 3d boundaries onto the 2d projection plane
        line([dbIS{iD}.vertices(dbIS{iD}.edges(:,1),1), dbIS{iD}.vertices(dbIS{iD}.edges(:,2),1)]',... 
             [dbIS{iD}.vertices(dbIS{iD}.edges(:,1),2), dbIS{iD}.vertices(dbIS{iD}.edges(:,2),2)]',...
             [dbIS{iD}.vertices(dbIS{iD}.edges(:,1),3), dbIS{iD}.vertices(dbIS{iD}.edges(:,2),3)]', ...
             'Color',myCol(iD,1:3),'LineWidth',1.5); hold on;
    end
end
a = minmax(1);  b = minmax(2);
line([a b; a a; a b; a a;   a a; a a; b b; b b;   a b; a a; a b; a a]', ...
     [a a; a b; b b; a b;   a a; b b; a a; b b;   a a; a b; b b; a b]', ...
     [b b; b b; b b; b b;   a b; a b; a b; a b;   a a; a a; a a; a a]', 'Color',.7*[1 1 1]);
axis square; camproj perspective; grid on; axis([minmax minmax minmax]); view([-25 15]); camlight left; lighting phong;
xlabel('r_1^{hat}'); ylabel('r_2^{hat}'); zlabel('r_3^{hat}'); set(gca,'XTick',-100:100,'YTick',-100:100,'ZTick',-100:100);
