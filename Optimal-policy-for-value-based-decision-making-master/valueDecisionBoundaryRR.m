function valueDecisionBoundaryRR()
% function valueDecisionBoundaryRR()
%
% This code generates the figures presented in the paper by Tajima, Drugowitsch & Pouget (2016) [1] and runs some extended simulations.
% 
% CITATION:
% [1] Satohiro Tajima*, Jan Drugowitsch*, and Alexandre Pouget.
% Optimal policy for value-based decision-making. 
% Nature Communications, 7:12400, (2016). 
% *Equally contributed.

global gamm geometric;
linearUtility = true; % (JARM 9th May '19) linear utility? (saturating otherwise)
geometric = true;   % (JARM 9th May '19) geometric discounting? (reward averaging otherwise)
gamm = 0.99;    % (JARM 9th May '19) geometric discount factor for future rewards 
tic;
Smax = 10;      % Grid range of states space (now we assume: S = [(Rhat1+Rhat2)/2, (Rhat1-Rhat2)/2]); Rhat(t) = (varR*X(t)+varX)/(t*varR+varX) )
resS = 801;      % Grid resolution of state space
tmax = 3;       % Time limit
dt   = .05;       % Time step
c    = 0.1;       % Cost of evidene accumulation
tNull = .25;     % Non-decision time + inter trial interval
g{1}.meanR = 2; % Prior mean of state (dimension 1)
g{1}.varR  = 5; % Prior variance of stte
g{1}.varX  = 2; % Observation noise variance
g{2}.meanR = 2; % Prior mean of state (dimension 2)
g{2}.varR  = 5; % Prior variance of state
g{2}.varX  = 2; % Observation noise variance
t = 0:dt:tmax;
Sscale = linspace(-Smax,Smax,resS);
[S{1},S{2}] = meshgrid(Sscale, Sscale);
iS0 = [findnearest(g{1}.meanR, Sscale) findnearest(g{2}.meanR, Sscale)];
Slabel = {'r_1^{hat}', 'r_2^{hat}'};

%% Utility functions:
if linearUtility
    utilityFunction = @(x) x;               % Linear utility function (for Fig. 3)
else
    utilityFunction = @(x) tanh(x);       % Saturating utility function (for Fig. 6)
end

%% Reward rate, Average-adjusted value, Decision:
Rh{1} = utilityFunction(S{1});                                                                              % Expected reward for option 1
Rh{2} = utilityFunction(S{2});                                                                              % Expected reward for option 2
RhMax = max_({Rh{1}, Rh{2}});                                                                               % Expected reward for decision
if geometric == false
    rho_ = fzero(@(rho) backwardInduction(rho,c,tNull,g,Rh,S,t,dt,iS0), g{1}.meanR, optimset('MaxIter',10));    % Reward rate
else
    rho_ = 0;  % (JARM 9th May '19) reward rate optimisation does not currently converge for geometric discounting
end
[V0, V, D, EVnext, rho, Ptrans, iStrans] = backwardInduction(rho_,c,tNull,g,Rh,S,t,dt,iS0);                 % Average-adjusted value, Decision, Transition prob. etc.
dbS2 = detectBoundary(D,S,t);

%% Transform to the space of accumulated evidence:
dbX = transformDecBound(dbS2,Sscale,t,g);

%% Run set of tests to measure value-sensitivity in equal alternative case
savePlots = true;
updateMeanVarianceEachStep=true; % if true, it uses the posterior as prior mean and variance in the next dt observation
computeDecisionBoundaries = false; % if true the code will compute the decision boundaries, if false will try to load the decision boundaries from the directory rawData (if it fails, it will recompute the data)
%suffix='-fixP';
suffix='-varP';
meanValues = 2:0.5:5; % mean reward values to be tested 
meanValues = 3.5:0.5:3.5; % mean reward values to be tested 
numruns = 1; % number of simulations per test case
allResults = zeros(length(meanValues)*numruns, 3); % structure to store the simulation results
j=1; % result line counter

Rh{1} = utilityFunction(S{1}); % Expected reward for option 1
Rh{2} = utilityFunction(S{2}); % Expected reward for option 2
for meanValue = meanValues
    g{1}.meanR = meanValue; g{2}.meanR = meanValue; % set prior with equal true mean for both options
    %g{1}.meanR = 3.5; g{2}.meanR = 3.5; % fix prior of mean to 3.5
    option1Mean = meanValue; option2Mean = meanValue; % set actual mean, from which the evidence data is drawn, equal for both options
    fprintf('mean value: %d \n',meanValue);
    if geometric
        filename = strcat('rawData/D_geom-',num2str(geometric),'_rm-',num2str(g{1}.meanR),'_S-',num2str(Smax),'-',num2str(resS),'_g-',num2str(gamm),'_t-',num2str(tmax),'_dt-',num2str(dt),'.mat');
    else
        filename = strcat('rawData/D_geom-',num2str(geometric),'_rm-',num2str(g{1}.meanR),'_S-',num2str(Smax),'-',num2str(resS),'_c-',num2str(c),'_t-',num2str(tmax),'_dt-',num2str(dt),'.mat');
    end
    dataLoaded = false;
    if ~computeDecisionBoundaries % load the decision threshold for all timesteps (matrix D)
        try
          load(filename, 'D','rho_');
          dataLoaded = true;
        catch
          disp('Could not load the decision matrix. Recomputing the data.'); 
        end
    end
    if ~dataLoaded % compute the decision threshold for all timesteps
        rho_ = 0; % we assume single decisions
%         iS0 = [findnearest(g{1}.meanR, Sscale) findnearest(g{2}.meanR, Sscale)]; % this line is not really needed
%         if geometric == false
%             rho_ = fzero(@(rho) backwardInduction(rho,c,tNull,g,Rh,S,t,dt,iS0), g{1}.meanR, optimset('MaxIter',10)); % Reward rate
%         else
%             rho_ = 0;  % reward rate optimisation does not currently converge for geometric discounting
%         end
        fprintf('computing boundaries...');
        [V0, V, D, EVnext, rho, Ptrans, iStrans] = backwardInduction(rho_,c,tNull,g,Rh,S,t,dt,iS0);
        fprintf('saving boundaries to file...');
        save(filename,'D','rho_', '-v7.3');
        fprintf('done.\n');
    end
    if savePlots % prepare the figure
        figure();
        clf;
        set(gcf, 'PaperUnits', 'inches');
        set(gcf, 'PaperSize', [8 1+numruns*1.5]);
        set(gcf, 'PaperPositionMode', 'manual');
        set(gcf, 'PaperPosition', [0 0 8 1+numruns*1.5]);
    end
    for run = 1:numruns
        if updateMeanVarianceEachStep
%             lastObs1 = 0; % last observation opt 1
%             lastObs2 = 0; % last observation opt 2
            r1m = g{1}.meanR; % mean opt 1
            r2m = g{2}.meanR; % mean opt 2
            r1v = g{1}.varR; % var opt 1
            r2v = g{2}.varR; % var opt 2
        end
        r1sum=0; % sum of observations opt 1
        r2sum=0; % sum of observations opt 2
        simTraj = [ ];
        for iT = 1:1:length(t)-1
            if updateMeanVarianceEachStep
%                 r1v = (r1v * g{1}.varX) / ( g{1}.varX + dt * r1v ); % compute the posterior variance for op 1
%                 r2v = (r2v * g{2}.varX) / ( g{2}.varX + dt * r2v ); % compute the posterior variance for op 2
%                 r1m = r1v * ( (r1m/r1v) + (lastObs1/g{1}.varX) ); % compute the posterior mean 1
%                 r2m = r2v * ( (r2m/r2v) + (lastObs2/g{2}.varX) ); % compute the posterior mean 2
                r1v = (r1v * g{1}.varX) / ( g{1}.varX + t(iT) * r1v ); % compute the posterior variance for op 1
                r2v = (r2v * g{2}.varX) / ( g{2}.varX + t(iT) * r2v ); % compute the posterior variance for op 2
                r1m = r1v * ( (r1m/r1v) + (r1sum/g{1}.varX) ); % compute the posterior mean 1
                r2m = r2v * ( (r2m/r2v) + (r2sum/g{2}.varX) ); % compute the posterior mean 2
                fprintf('At time %d means, %d,%d and variances: %d,%d.\n',t(iT),r1m,r2m,r1v,r2v)
            else
                r1m = ( (g{1}.meanR * g{1}.varX) + (r1sum * g{1}.varR) ) / (g{1}.varX + t(iT) * g{1}.varR ); % compute the posterior mean 1
                r2m = ( (g{2}.meanR * g{2}.varX) + (r2sum * g{2}.varR) ) / (g{2}.varX + t(iT) * g{2}.varR ); % compute the posterior mean 2
            end
            if savePlots 
                simTraj = [simTraj; r1m r2m ];
            end
            % find the index of the posterior mean to check in matrix D if decision is made
            r1i = findnearest(r1m, Sscale, -1);
            r2i = findnearest(r2m, Sscale, -1);
            %fprintf('For values (%d,%d) at time %d the D is %d\n',r1m,r2m, t(iT), D(r1i,r2i,iT) )
            try
                decisionMade = D(r1i,r2i,iT)==1 || D(r1i,r2i,iT)==2 || D(r1i,r2i,iT)==1.5;
            catch
                fprintf('ERROR for D=%d, t=%d, r1m=%d, r2m=%d, r1i=%d, r2i=%d,\n',D(r1i,r2i,iT),iT,r1m,r2m,r1i,r2i);
            end
            if decisionMade
                break
            else
%                 lastObs1 = normrnd(option1Mean*dt, sqrt(g{1}.varX*dt)); % last observation opt 1
%                 lastObs2 = normrnd(option2Mean*dt, sqrt(g{2}.varX*dt)); % last observation opt 2
                r1sum = r1sum + normrnd(option1Mean*dt, sqrt(g{1}.varX*dt)); % add the last observation opt 1 to the sum
                r2sum = r2sum + normrnd(option2Mean*dt, sqrt(g{2}.varX*dt)); % add the last observation opt 2 to the sum
            end
        end
        if savePlots
            subplot(ceil(numruns/2),2,run); imagesc(Sscale, Sscale, D(:,:,iT), [1 3]); axis square; axis xy; title(['D=' num2str(D(r1i,r2i,iT)) ' t=' num2str(t(iT)) ]); xlabel(Slabel{1}); ylabel(Slabel{2});
            hold on; plot(simTraj(:,1),simTraj(:,2),'r','linewidth',2);
            hold on; plot(r1m,r2m,'wo','linewidth',2);
            filename = strcat('rawData/traj_geometric-',num2str(geometric),'_r1-',num2str(option1Mean),'_r2-',num2str(option2Mean),'_',num2str(run),'.txt');
            csvwrite(filename,simTraj);
        end
        allResults(j,:) = [ meanValue D(r1i,r2i,iT) t(iT) ];
        j = j+1;
    end
    if savePlots
        filename = strcat('simFigs/geometric-',num2str(geometric),'_pm-',num2str(g{1}.meanR),'_rm-',num2str(meanValue),'.pdf');
        saveas(gcf,filename)
    end
end
filename = strcat('resultsData/vs_geometric-',num2str(geometric),suffix,'.txt');
csvwrite(filename,allResults);

%% Plot value sensitive values
addpath('../plottinglibs/');
suffix='-fixP'; %  suffix='-varP';
figure();
clf;
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [8 4]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 8 4]);
for geo = [true false]
    filename = strcat('resultsData/vs_geometric-',num2str(geo),suffix,'.txt');
    allResults = readtable(filename);
    fprintf('Loaded file %s with %d lines.\n', filename, height(allResults)/length(meanValues));
    dataForPlot = [];
    for meanValue = meanValues
        dataForPlot = [ dataForPlot, allResults{ allResults{:,1} == meanValue, 3}];
    end
    if geo == true
        geoTitle = 'geometric cost ';
    else
        geoTitle = 'linear cost ';
    end
    if suffix == '-fixP'
        priorTitle = '- constant prior -';
    else
        priorTitle = '- variable prior -';
    end
    subplot(2,2,1+geo); errorbar(meanValues, mean(dataForPlot), std(dataForPlot)/sqrt(height(allResults)/length(meanValues))*1.96); title([geoTitle priorTitle ' mean']);
    %violin(dataForPlot,'xlabel',string(meanValues))
    subplot(2,2,3+geo); distributionPlot(dataForPlot,'histOpt',1,'colormap',spring,'xValues',meanValues); title([geoTitle priorTitle ' all data']);
end
filename = strcat('simFigs/value-sensitive',suffix,'.pdf');
saveas(gcf,filename)

%% Simulate and plot results
numruns=1;

figure();
clf;
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [8 numruns*2]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 8 numruns*2]);

for run = 1:numruns
    r1sum=0;
    r2sum=0;
    simTraj = [ ];
    for iT = 1:1:length(t)-1
        r1m = ( (g{1}.meanR * g{1}.varX) + (r1sum * g{1}.varR) ) / (g{1}.varX + t(iT) * g{1}.varR );
        r2m = ( (g{2}.meanR * g{2}.varX) + (r2sum * g{2}.varR) ) / (g{2}.varX + t(iT) * g{2}.varR );
        simTraj = [simTraj; r1m r2m ];
        r1i = findnearest(r1m, Sscale, -1);
        r2i = findnearest(r2m, Sscale, -1);
        %fprintf('For values (%d,%d) at time %d the D is %d\n',r1v,r2v, t(iT), D(r1i,r2i,iT) )
        if D(r1i,r2i,iT)==1 || D(r1i,r2i,iT)==2 || D(r1i,r2i,iT)==1.5
            break
        else
            r1sum = r1sum + normrnd(g{1}.meanR*dt, sqrt(g{1}.varX*dt));
            r2sum = r2sum + normrnd(g{2}.meanR*dt, sqrt(g{2}.varX*dt));
        end
    end
    subplot(ceil(numruns/2),2,run); imagesc(Sscale, Sscale, D(:,:,iT), [1 3]); axis square; axis xy; title(['D(0) \rho=' num2str(rho_,3)]); xlabel(Slabel{1}); ylabel(Slabel{2});
    hold on; plot(simTraj(:,1),simTraj(:,2),'r','linewidth',2);
    hold on; plot(r1m,r2m,'wo','linewidth',2);
    %filename = strcat('simFigs/geometric-',num2str(geometric),'_r',num2str(run),'.pdf');
end
filename = strcat('simFigs/geometric-',num2str(geometric),'_r1-',num2str(g{1}.meanR),'_r2-',num2str(g{2}.meanR),'.pdf');
%saveas(gcf,filename)

%% Plot intermediate steps
stepsize = 30;
timesteps = stepsize:stepsize:length(simTraj);
if timesteps(length(timesteps))~=length(simTraj) timesteps=[timesteps length(simTraj)]; end
figure();
clf;
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [8 length(timesteps)*2]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 8 length(timesteps)*2]);
for iT = timesteps
    subplot(ceil(length(timesteps)/2),2,findnearest(iT,timesteps));
    imagesc(Sscale, Sscale, D(:,:,iT), [1 3]); axis square; axis xy; 
    title(['time=' num2str(t(iT)) ]);
    hold on; plot(simTraj(1:iT,1),simTraj(1:iT,2),'r','linewidth',2);
    hold on; plot(simTraj(iT,1),simTraj(iT,2),'wo','linewidth',2);
end
filename = strcat('simFigs/fullSim-geometric-',num2str(geometric),'_r1-',num2str(g{1}.meanR),'_r2-',num2str(g{2}.meanR),'.pdf');
saveas(gcf,filename)


%% Giovanni's plots
figure;
iS2 = findnearest(0.5, Sscale, -1);
rect = [-Smax Smax -1 1 -2.3 .5];
time=400;
[r1Max,r2Max,vMax] = plotSurf(Sscale, V(:,:,time), iS2, [0 0 0], Slabel); axis(rect); title('V(0)');
clf;
if geometric == false
    [r1Acc,r2Acc,vAcc] = plotSurf(Sscale, EVnext(:,:,time)-(rho+c)*dt, iS2, [1 0 0], Slabel); axis(rect); title('<V(\deltat)|R^{hat}(0)> - (\rho+c)\deltat');
else
    [r1Acc,r2Acc,vAcc] = plotSurf(Sscale, EVnext(:,:,time)-(rho)*dt, iS2, [1 0 0], Slabel); axis(rect); title('<V(\deltat)|R^{hat}(0)> - (\rho+c)\deltat');
end
[r1Dec,r2Dec,vDec] = plotSurf(Sscale, RhMax-rho*tNull         , iS2, [0 0 1], Slabel); axis(rect); title('max(R_1^{hat},R_2^{hat}) - \rho t_{Null}');
    
iS2 = findnearest(1, Sscale, -1);
if geometric == false
    [r1Acc2,r2Acc2,vAcc2] = plotSurf(Sscale, EVnext(:,:,time)-(rho+c)*dt, iS2, [1 0 0], Slabel); axis(rect); title('<V(\deltat)|R^{hat}(0)> - (\rho+c)\deltat');
else
    [r1Acc2,r2Acc2,vAcc2] = plotSurf(Sscale, EVnext(:,:,time)-(rho)*dt, iS2, [1 0 0], Slabel); axis(rect); title('<V(\deltat)|R^{hat}(0)> - (\rho+c)\deltat');
end
[r1Dec2,r2Dec2,vDec2] = plotSurf(Sscale, RhMax-rho*tNull         , iS2, [0 0 1], Slabel); axis(rect); title('max(R_1^{hat},R_2^{hat}) - \rho t_{Null}');

clf;
plot((r1Max-r2Max)/2, vMax, 'k:', (r1Acc-r2Acc)/2, vAcc, 'r', (r1Dec-r2Dec)/2, vDec, 'b'); xlabel(['(' Slabel{1} '-' Slabel{2} ')/2']); xlim(rect(1:2));
hold on;
plot((r1Max-r2Max)/2, vMax, 'k:', (r1Acc2-r2Acc2)/2, vAcc2, 'r', (r1Dec2-r2Dec2)/2, vDec2, 'b'); xlabel(['(' Slabel{1} '-' Slabel{2} ')/2']); xlim(rect(1:2));
%clf;
%plot(r1Max, vMax, 'k:', r1Acc, vAcc, 'r', r1Dec, vDec, 'b'); xlabel([ Slabel{1} ]); xlim([-Smax Smax]);
figure();
imagesc(Sscale, Sscale, D(:,:,time), [1 3]); axis square; axis xy; title(['D(0) \rho=' num2str(rho_,3)]); xlabel(Slabel{1}); ylabel(Slabel{2}); 
r1v=0.5; r2v=0;
r1i = findnearest(r1v, Sscale, -1);
r2i = findnearest(r2v, Sscale, -1);
hold on;
plot(r1v,r2v,'ro','linewidth',2)
fprintf('For values (%d,%d) the D is %d\n',r1v,r2v, D(r1i,r2i,time) )
% hold on; axis(rect(1:4)); plot(r1Max, r2Max, 'r-');
% figure();
% plotSurf(Sscale, RhMax-rho*tNull - (EVnext(:,:,1)-(rho+c)*dt), iS2, [0 1 0], Slabel); xlim(rect(1:2)); ylim(rect(1:2));

%% - Show results -
figure; clf; colormap bone;
iS2 = findnearest(.5, Sscale, -1);
iTmax = length(t);
rect = [-1 1 -1 1 -2.3 .5];

%% t=0:
subplotXY(5,4,2,1); [r1Max,r2Max,vMax] = plotSurf(Sscale, V(:,:,1)                , iS2, [0 0 0], Slabel); axis(rect); title('V(0)');
%                     plot3(g{1}.meanR, g{2}.meanR, V0, 'g.', 'MarkerSize',15);
subplotXY(5,4,3,1); [r1Acc,r2Acc,vAcc] = plotSurf(Sscale, EVnext(:,:,1)-(rho+c)*dt, iS2, [1 0 0], Slabel); axis(rect); title('<V(\deltat)|R^{hat}(0)> - (\rho+c)\deltat');
subplotXY(5,4,4,1); [r1Dec,r2Dec,vDec] = plotSurf(Sscale, RhMax-rho*tNull         , iS2, [0 0 1], Slabel); axis(rect); title('max(R_1^{hat},R_2^{hat}) - \rho t_{Null}');
subplotXY(5,4,5,1); hold on;
    plot((r1Max-r2Max)/2, vMax, 'k:', (r1Acc-r2Acc)/2, vAcc, 'r', (r1Dec-r2Dec)/2, vDec, 'b');
    xlabel(['(' Slabel{1} '-' Slabel{2} ')/2']); xlim(rect(1:2));
subplotXY(5,4,1,1); imagesc(Sscale, Sscale, D(:,:,  1), [1 3]); axis square; axis xy;
    title(['D(0) \rho=' num2str(rho_,3)]); xlabel(Slabel{1}); ylabel(Slabel{2}); hold on; axis(rect(1:4));
                    plot(r1Max, r2Max, 'r-');
%                     plot(g{1}.meanR, g{2}.meanR, 'g.');

%% t=0 (superimposed & difference):
subplotXY(5,4,3,2); plotSurf(Sscale, EVnext(:,:,1)-(rho+c)*dt, iS2, [1 0 0], Slabel); hold on;
                    plotSurf(Sscale, RhMax-rho*tNull         , iS2, [0 0 1], Slabel); axis(rect);
subplotXY(5,4,4,2); plotSurf(Sscale, RhMax-rho*tNull - (EVnext(:,:,1)-(rho+c)*dt), iS2, [0 1 0], Slabel); xlim(rect(1:2)); ylim(rect(1:2));

%% t=dt:
subplotXY(5,4,2,2); plotSurf(Sscale, V(:,:,2),      iS2, [0 0 0], Slabel); axis(rect); title('V(\deltat)');
subplotXY(5,4,1,2); imagesc(Sscale, Sscale, D(:,:,  2), [1 3]); axis square; axis xy; title('D(\deltat)'); xlabel(Slabel{1}); ylabel(Slabel{2}); hold on; axis(rect(1:4));

%% t=T-dt:
subplotXY(5,4,1,3); imagesc(Sscale, Sscale, D(:,:,iTmax-1), [1 3]); axis square; axis xy;
    title('D(T-\deltat)'); hold on; rectangle('Position',[rect(1) rect(3) rect(2)-rect(1) rect(4)-rect(3)]); axis(rect);
subplotXY(5,4,2,3); [r1Max,r2Max,vMax] = plotSurf(Sscale, V(:,:,iTmax-1)                , iS2, [0 0 0], Slabel); axis(rect); title('V(T-\deltat)')
subplotXY(5,4,3,3); [r1Acc,r2Acc,vAcc] = plotSurf(Sscale, EVnext(:,:,iTmax-1)-(rho+c)*dt, iS2, [1 0 0], Slabel); axis(rect); title('<V(T)|R^{hat}(T-\deltat)> - (\rho+c) \deltat');
subplotXY(5,4,4,3); [r1Dec,r2Dec,vDec] = plotSurf(Sscale, RhMax-rho*tNull               , iS2, [0 0 1], Slabel); axis(rect); title('max(R_1^{Hat},R_2^{Hat}) - \rho t_{Null}');
subplotXY(5,4,5,3); hold on;
    plot((r1Max-r2Max)/2, vMax, 'k:', (r1Acc-r2Acc)/2, vAcc, 'r', (r1Dec-r2Dec)/2, vDec, 'b');
    xlabel(['(' Slabel{1} '-' Slabel{2} ')/2']); xlim(rect(1:2));
    
%% t=T:
subplotXY(5,4,1,4); imagesc(Sscale, Sscale, D(:,:,iTmax), [1 3]); axis square; axis xy; title('D(T)'); hold on; axis(rect(1:4));
subplotXY(5,4,2,4); plotSurf(Sscale, V(:,:,iTmax), iS2, [0 0 0], Slabel); title('V(T) = max(R_1^{hat},R_2^{hat}) - \rho t_{Null}'); axis(rect);

toc;

%% write D timelapse video to file (JARM 21st May '19)
v = VideoWriter('D.avi');
open(v);

figure;
imagesc(Sscale, Sscale, D(:,:,iTmax), [1 3]); axis square; axis xy; title('D(T)'); hold on; axis(rect(1:4));
set(gca,'nextplot','replacechildren');
for i = 1:iTmax
  imagesc(Sscale, Sscale, D(:,:,i), [1 3]); axis square; axis xy; title('D(T)'); hold on; axis(rect(1:4));
  frame = getframe;
  writeVideo(v,frame);
end    
close(v);

function [V0, V, D, EVnext, rho, Ptrans, iStrans] = backwardInduction(rho_,c,tNull,g,Rh,S,t,dt,iS0)
global gamm geometric;
k = 0;                                                                        % Reward rate estimate
rho = k*S{1}/tNull + (1-k)*rho_;
if geometric
    [V(:,:,length(t)), D(:,:,length(t))] = max_({Rh{1}-rho*tNull, Rh{2}-rho*tNull});                        % Max V~ at time tmax
else
    [V(:,:,length(t)), D(:,:,length(t))] = max_({Rh{1}-rho*tNull, Rh{2}-rho*tNull});                        % Max V~ at time tmax
end
for iT = length(t)-1:-1:1
    [EVnext(:,:,iT), Ptrans{iT}, iStrans{iT}] = E(V(:,:,iT+1),S,t(iT),dt,g);                            % <V~(t+1)|S(t)> for waiting
%    disp(size(Rh{1}));
%    disp(size(Rh{2}));
%    disp(size(EVnext(:,:,iT)));
    if geometric
        [V(:,:,iT), D(:,:,iT)] = max_({Rh{1}-rho*tNull, Rh{2}-rho*tNull, EVnext(:,:,iT)*gamm});                                % (JARM 9th May '19) [geometrically-discounted value (V~), decision] at time t    
    else
        [V(:,:,iT), D(:,:,iT)] = max_({Rh{1}-rho*tNull, Rh{2}-rho*tNull, EVnext(:,:,iT)-(rho+c)*dt});       % [Average-adjusted value (V~), decision] at time t
    end
    
%     fprintf('%d/%d\t',iT,length(t)-1); toc;
end
disp(iS0(1));
%V0 = mean(vector(V(iS0(1),iS0(2),1)));
V0 = V(iS0(1),iS0(2),1); % JARM (8th May '19)
fprintf('rho = %d\tV0 = %d\t', rho_, V0); toc;

function R = extrap(mat, varargin)
    % JARM (9th May '19) original function missing; identity function used and code fixed around this
    R = mat;

function [EV, Ptrans, iStrans] = E(V,S,t,dt,g)
g{1}.varRh = g{1}.varR * g{1}.varX / (t * g{1}.varR + g{1}.varX);
g{2}.varRh = g{2}.varR * g{2}.varX / (t * g{2}.varR + g{2}.varX);
v1 = varTrans(g{1}.varRh, g{1}.varR, g{1}.varX, t, dt);
v2 = varTrans(g{2}.varRh, g{2}.varR, g{2}.varX, t, dt);
aSscale = abs(S{1}(1,:));
iStrans{1} = find(aSscale<3*sqrt(v1));
iStrans{2} = find(aSscale<3*sqrt(v2));
Ptrans = normal2({S{1}(iStrans{2},iStrans{1}),S{2}(iStrans{2},iStrans{1})}, [0 0], [v1 0; 0 v2]);
mgn = ceil(size(Ptrans)/2);
V = extrap(V,mgn,[5 5]); % JARM (8th May '19) ???
EV = conv2(V,Ptrans,'same'); % JARM (8th May '19) marginalise expected value over probabilities of future states
%EV = EV(mgn(1)+1:end-mgn(1), mgn(2)+1:end-mgn(2)); % JARM (8th May '19) select central sub-region of larger expected value array

function v = varTrans(varRh, varR, varX, t, dt)
% v = (varR * (varX + varRh)) / ((1 + t/dt) * varR + varX / dt);
v = (varR ./ (varR*(t+dt) + varX)).^2 .* (varX + varRh * dt) * dt;

function prob = normal2(x, m, C)
d1 = x{1} - m(1);
d2 = x{2} - m(2);
H = -1/2*(C\eye(2)); prob = exp(bsxfun(@plus,d1.*d1*H(1,1), d1.*d2*H(1,2)) + d2.*d1*H(2,1) + d2.*d2*H(2,2));
% prob = exp(-(d1.^2/C(1,1)/2 + d2.^2/C(2,2))/2);
prob = prob ./ sum(prob(:));

function [V, D] = max_(x)
x_ = zeros(size(x{1},1), size(x{1},2), length(x));
for k = 1:length(x)
    x_(:,:,k) = x{k};
end
[V, D] = max(x_,[],3);
D(x{1}==x{2} & D==1) = 1.5;

function dbS2 = detectBoundary(D,S,t)
dS = diff(S{2}(1:2,1));
S_ = repmat(S{2},[1 1 length(t)]); S_(D~=1 & D~=1.5) =  Inf; dbS2(:,:,1) = max(squeeze(min(S_))-dS, 0);                % Decision boundary [min(S2;dec=1); max(S2;dec=2)]
S_ = repmat(S{2},[1 1 length(t)]); S_(D~=2 & D~=1.5) = -Inf; dbS2(:,:,2) = min(squeeze(max(S_))+dS, 0);                %  ... bndS2(iS1, iTime, iDec)
mgn = 1; [sm{1},sm{2}] = meshgrid(-mgn:mgn,-mgn:mgn);
for k=1:2
    %% Extrapolating:
    db_ = dbS2(:,:,k); db_(~isfinite(db_) & isfinite([db_(:,2:end) db_(:,end)])) = (-1)^(k+1)*max(max(S{1}));  dbS2(:,:,k) = db_; % JARM (8th May '19) changed vector() call to max()
    
    %% Smoothing:
    db_ = conv2(extrap(dbS2(:,:,k),mgn),normal2(sm,[0 0],[1 0; 0 1]),'same'); 
%    dbS2(:,:,k) = db_(mgn+1:end-mgn,mgn+1:end-mgn);
    dbS2(:,:,k) = db_;     % JARM (8th May '19)
end

function [dbX, dbR] = transformDecBound(dbS2,Sscale,t,g)
S1 = repmat(Sscale',[1 size(dbS2,2) size(dbS2,3)]);
t_ = repmat(t,[size(dbS2,1) 1 size(dbS2,3)]);
for k=1:2;  mR{k}=g{k}.meanR;  vR{k}=g{k}.varR;  vX{k}=g{k}.varX;  end
dbX(:,:,:,1) = (t_+(vX{1}+vX{2})./(vR{1}+vR{2})) .* (S1+dbS2) - (vX{1}+vX{2})./(vR{1}+vR{2}) .* (mR{1}+mR{2});          % X1 (iS1, iTime, iDec, 1)
dbX(:,:,:,2) = (t_+(vX{1}+vX{2})./(vR{1}+vR{2})) .* (S1-dbS2) - (vX{1}+vX{2})./(vR{1}+vR{2}) .* (mR{1}-mR{2});          % X2 (iS1, iTime, iDec, 2)
dbR(:,:,:,1) = (S1+dbS2);          % R1 (iS1, iTime, iDec, 1)
dbR(:,:,:,2) = (S1-dbS2);          % R2 (iS1, iTime, iDec, 2)

function [x_,y_,v_] = plotSurf(Sscale, Val, iS, col, Slabel)
[x,y] = meshgrid(1:length(Sscale), 1:length(Sscale));
x_ = Sscale(x(x+y==iS+round(length(Sscale)/2)));
y_ = Sscale(y(x+y==iS+round(length(Sscale)/2)));
v_ = Val(x+y==iS+round(length(Sscale)/2));
h = surfl(Sscale, Sscale, Val); hold on; %camproj perspective;
set(h,'FaceColor', col, 'EdgeColor','none'); camlight left; lighting phong; alpha(0.7) % JARM (8th May '19) replaced sat(.5,col) with col
if ischar(col);  plot3(x_, y_, v_,         col); hold on;
else             plot3(x_, y_, v_, 'Color',col); hold on;  end
xlabel(Slabel{1}); ylabel(Slabel{2}); %zlim([-50 50]);

