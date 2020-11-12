function numericalSimulations(util, geom, gam, maxv, logs)
addpath('../shared')
global gamm geometric epsil; % (JARM 23rd August '19)
global valscale; % (JARM 7th October '19)
utility = util;
gamm = gam;
geometric = geom;
maxval = maxv;
logslope = logs;
fprintf("Script started with utility %s (maxval %1.2f, logslope %1.2f) and geometric %d\n",utility, maxval, logslope, geometric)
%geometric = false; % (JARM 23rd August '19) use geometric discounting for future rewards 
%gamm = 0.6; % (JARM 23rd August '19) geometric discount factor for future rewards 
%gamm = 0.01; % Original value
epsil = 0; % (JARM 11th September '19) epsilon error to add to co-planar services to compute convex hull (required to check geometric discounting results; deprecated)
valscale = 0.5; % (JARM 7th October '19) move triangle along diagonal as option values scale)
%maxval = 4; % (JARM 6th March '20) maximum utility for logistic utility function
%utility='linear';
tic;
Smax = 8;      % Grid range of states space (now we assume: S = [(Rhat1+Rhat2)/2, (Rhat1-Rhat2)/2]); Rhat(t) = (varR*X(t)+varX)/(t*varR+varX) )
%Smax = 4;      % Original value
resSL  = 15;      % Grid resolution of state space
resS = 151;      % Grid resolution of state space
%resS = 101;      % Original value
tmax = 37;       % Time limit
%tmax = 3;       % Original value
dt   = .0625;      % Time step for computing the decision matrix
%dt   = .05;       % Original value
dtsim   = .005;    % Time step for computing the stochastic simulations
c    = 0.6;       % Cost of evidence accumulation
%tNull = 1;     % Non-decision time + inter trial interval
meanValue = 1.5;
g{1}.meanR = meanValue; % Prior mean of state (dimension 1)
g{1}.varR  = 5; % Prior variance of stte
g{1}.varX  = 2; % Observation noise variance
g{2}.meanR = meanValue; % Prior mean of state (dimension 2)
g{2}.varR  = 5; % Prior variance of state
g{2}.varX  = 2; % Observation noise variance
g{3}.meanR = meanValue; % Prior mean of state (dimension 3)
g{3}.varR  = 5; % Prior variance of state
g{3}.varX  = 2; % Observation noise variance
t = 0:dt:tmax;
Slabel = {'r_1^{hat}', 'r_2^{hat}', 'r_3^{hat}'};
myCol = [1 0 0; 0 1 0; 0 0 1];

if contains(utility , 'all')
    utilities=["linear", "sqrt", "logLm"] %, "logHm", "tan"];
else
    utilities=utility;
end

%% Run set of tests to measure value-sensitivity in equal alternative case
for utility = utilities
%for utility = ["linear", "logLm", "logHm", "sqrt", "tan"] 
%for geo = [false true]
%for geo = true
%if geo; geometric=true; else; geometric=false; end
savePlots = false; % set true only for few runs (e.g. 6)
singleDecisions=false; % if true we model single decision (i.e. expected future rewards=0); if false we compute the expected future reward (rho_)
meanValues=0.5:0.5:3; % mean reward values to be tested
meanValues = -1:0.1:3; % mean reward values to be tested 
numruns=10000; % number of simulations per test case
allResults=zeros(length(meanValues)*numruns, 3); % structure to store the simulation results
j=1; % result line counter

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
suffix=[singleDecisionsSuffix '_u-' utility];

Sscale = linspace(-Smax, Smax, resS); % define the range of possible rewards
dataLoaded = false;
for meanValue = meanValues    
    option1Mean = meanValue; option2Mean = meanValue; option3Mean = meanValue; % set actual mean, from which the evidence data is drawn, equal for both options
    if geometric
        filename = strcat('rawData/D_geom-',num2str(geometric),'_mv-',num2str(maxval),'_ls-',num2str(logslope),'_rm-',num2str(g{1}.meanR),'_S-',num2str(Smax),'-',num2str(resS),'_g-',num2str(gamm),'_t-',num2str(tmax),strjoin(suffix,''),'.mat');
    else
        filename = strcat('rawData/D_geom-',num2str(geometric),'_mv-',num2str(maxval),'_ls-',num2str(logslope),'_rm-',num2str(g{1}.meanR),'_S-',num2str(Smax),'-',num2str(resS),'_c-',num2str(c),'_t-',num2str(tmax),strjoin(suffix,''),'.mat');
    end
    if (~dataLoaded)
        fprintf('loading boundaries...');
        load(filename, 'D','rho_');
        dataLoaded = true;
        fprintf('done.\n');
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
        simTraj = [ ];
        for simT = 0:dtsim:tmax %iT = 1:1:length(t)-1
            iT = findnearest(simT, t);
            r1m = ( (g{1}.meanR * g{1}.varX) + (r1sum * g{1}.varR) ) / (g{1}.varX + t(iT) * g{1}.varR ); % compute the posterior mean 1
            r2m = ( (g{2}.meanR * g{2}.varX) + (r2sum * g{2}.varR) ) / (g{2}.varX + t(iT) * g{2}.varR ); % compute the posterior mean 2
            r3m = ( (g{3}.meanR * g{3}.varX) + (r3sum * g{3}.varR) ) / (g{3}.varX + t(iT) * g{3}.varR ); % compute the posterior mean 3
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
                r1sum = r1sum + normrnd(option1Mean*dtsim, sqrt(g{1}.varX*dtsim));
                r2sum = r2sum + normrnd(option2Mean*dtsim, sqrt(g{2}.varX*dtsim));
                r3sum = r3sum + normrnd(option3Mean*dtsim, sqrt(g{3}.varX*dtsim));
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
            filename = strcat('rawData/traj_geometric-',num2str(geometric),'_mv-',num2str(maxval),'_ls-',num2str(logslope),'_r1-',num2str(option1Mean),'_r2-',num2str(option2Mean),'_r3-',num2str(option3Mean),'_',num2str(run),'.txt');
            csvwrite(filename,simTraj);
        end
        dec=D(r1i,r2i,r3i,iT);
        if isempty(dec); dec=0; end
        allResults(j,:) = [ meanValue dec t(iT) ];
        j = j+1;
    end
    if savePlots
        if geometric
            filename = strcat('simFigs/geometric-',num2str(geometric),'_mv-',num2str(maxval),'_ls-',num2str(logslope),'_pm-',num2str(g{1}.meanR),'_rm-',num2str(meanValue),'_S-',num2str(Smax),'-',num2str(resS),'_g-',num2str(gamm),'_t-',num2str(tmax),strjoin(suffix,''),'.pdf');
        else
            filename = strcat('simFigs/geometric-',num2str(geometric),'_mv-',num2str(maxval),'_ls-',num2str(logslope),'_pm-',num2str(g{1}.meanR),'_rm-',num2str(meanValue),'_S-',num2str(Smax),'-',num2str(resS),'_c-',num2str(c),'_t-',num2str(tmax),strjoin(suffix,''),'.pdf'); 
        end
        saveas(gcf,filename)
    end
end
if geometric
    filename = strcat('resultsData/vs_geometric-',num2str(geometric),'_mv-',num2str(maxval),'_ls-',num2str(logslope),'_rm-',num2str(g{1}.meanR),'_S-',num2str(Smax),'-',num2str(resS),'_g-',num2str(gamm),'_t-',num2str(tmax),strjoin(suffix,''),'.txt');
else
    filename = strcat('resultsData/vs_geometric-',num2str(geometric),'_mv-',num2str(maxval),'_ls-',num2str(logslope),'_rm-',num2str(g{1}.meanR),'_S-',num2str(Smax),'-',num2str(resS),'_c-',num2str(c),'_t-',num2str(tmax),strjoin(suffix,''),'.txt');    
end
csvwrite(filename,allResults);
%end
end

