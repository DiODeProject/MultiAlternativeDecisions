function numericalSimulations()
addpath('../shared')
global gamm geometric epsil; % (JARM 23rd August '19)
global valscale; % (JARM 7th October '19)
geometric = false; % (JARM 23rd August '19) use geometric discounting for future rewards 
gamm = 0.6; % (JARM 23rd August '19) geometric discount factor for future rewards 
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

%% Run set of tests to measure value-sensitivity in equal alternative case
for utility = ["linear", "logLm", "sqrt"]
%for utility = ["linear", "logLm", "logHm", "sqrt", "tan"] 
for geo = [false true]
%for geo = true
if geo; geometric=true; else; geometric=false; end
savePlots = false; % set true only for few runs (e.g. 6)
singleDecisions=true; % if true we model single decision (i.e. expected future rewards=0); if false we compute the expected future reward (rho_)
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
        filename = strcat('rawData/D_geom-',num2str(geometric),'_rm-',num2str(g{1}.meanR),'_S-',num2str(Smax),'-',num2str(resS),'_g-',num2str(gamm),'_t-',num2str(tmax),singleDecisionsSuffix,'_u-',utility,'.mat');
    else
        filename = strcat('rawData/D_geom-',num2str(geometric),'_rm-',num2str(g{1}.meanR),'_S-',num2str(Smax),'-',num2str(resS),'_c-',num2str(c),'_t-',num2str(tmax),singleDecisionsSuffix,'_u-',utility,'.mat');
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
for utility = ["linear", "logLm", "sqrt"]%,"tan",  "logHm"]
meanValues = 0:0.1:3;
full='-full';full='';
addpath('../plottinglibs/');
suffix1=strcat('-singleDec_u-',utility,'_c-'); %suffix='-varP';
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
        dataForPlot = [ dataForPlot, allResults{ abs(allResults{:,1} - meanValue)<0.0000001, 3}];
    end
    if geo == true
        geoTitle = 'nonlinear time';
    else
        geoTitle = 'linear time';
    end
    subplot(1,2,1+geo); errorbar(meanValues, mean(dataForPlot), std(dataForPlot)/sqrt(height(allResults)/length(meanValues))*1.96,'LineWidth', 1.5);
    title(strcat(geoTitle,' - ',strrep(strrep(utility,'logLm','nonlinear (logistic)'),'sqrt','nonlinear (sqrt)'),' utility'));
    set(gca,'FontSize',13)
    % Fixing the axes
    ymin=0; ymax=0.55;
    epsilon=(max(meanValues)-min(meanValues))*0.05;
    axis([min(meanValues)-epsilon max(meanValues)+epsilon ymin ymax]);
    
    %violin(dataForPlot,'xlabel',string(meanValues))
    %subplot(2,2,3+geo); 
    %errorbar(meanValues, trimmean(dataForPlot,0.5), std(dataForPlot)/sqrt(height(allResults)/length(meanValues))*1.96); title(['3n - ' geoTitle priorTitle ' aa']);
    %distributionPlot(dataForPlot,'histOpt',1,'colormap',spring,'xValues',meanValues); title(['3n - ' geoTitle priorTitle ' all data']);
end
filename = strcat('simFigs/value-sensitive',full,strrep(suffix,'.',''),'.pdf');
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
        dataForPlot = [ dataForPlot, allResults{ abs(allResults{:,1} - meanValue)<0.0000001, 3}];
    end
    hold on; errorbar(meanValues, mean(dataForPlot), std(dataForPlot)/sqrt(height(allResults)/length(meanValues))*1.96,'LineWidth', 2.5);
    ymin = min(ymin, min(mean(dataForPlot)-std(dataForPlot)/sqrt(height(allResults)/length(meanValues))*3) );
    ymax = max(ymax, max(mean(dataForPlot)+std(dataForPlot)/sqrt(height(allResults)/length(meanValues))*3) );
end
ymin=0; ymax=0.55;
epsilon=(max(meanValues)-min(meanValues))*0.05;
axis([min(meanValues)-epsilon max(meanValues)+epsilon ymin ymax]);
pos='northeast';
if contains(utility , 'linear') || contains(utility , 'sqrt'); pos='southeast'; end
hleg = legend('Linear','Geometric','FontSize',13,'Location',pos);%,'Location','east')
htitle = get(hleg,'Title');
set(htitle,'String','Temporal discount')
title(strcat(strrep(strrep(utility,'logLm','nonlinear (logistic)'),'sqrt','nonlinear (sqrt)'),' utility'));
xlabel('Stimuli''s magnitude')
ylabel('Reaction time')
set(gca,'FontSize',13)
pbaspect([3 2 2])
filename = strcat('simFigs/vs-comp',full,strrep(suffix,'.',''),'.pdf');
saveas(gcf,filename);
end

%% Scatter plot
for utility = ["linear"]%, "logLm", "sqrt"]%, "logHm", "tan"]
meanValues = 0:0.1:2;
suffix1=strcat('-singleDec_u-',utility,'_c-'); %suffix='-varP';
figure();
clf;
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [8 4]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 8 4]);
ymin=1000; ymax=0;
for geo = [true ]
    if geo; cost=gamm; else; cost=c; end
    suffix=strjoin([suffix1 num2str(cost)],'');
    filename = strcat('resultsData/vs_geometric-',num2str(geo),suffix,'.txt');
    allResults = readtable(filename);
    fprintf('Loaded file %s with %d lines.\n', filename, height(allResults)/length(meanValues));
    for meanValue = meanValues
        dataForPlot = allResults{ abs(allResults{:,1} - meanValue)<0.0000001, 3};
        scatter(repmat(meanValue,length(dataForPlot),1), dataForPlot,'MarkerFaceColor','b','MarkerEdgeColor','b',...
    'MarkerFaceAlpha',.02,'MarkerEdgeAlpha',.02);
        hold on; 
        ymin = min(ymin, min(dataForPlot) );
        ymax = max(ymax, max(dataForPlot) );
    end
    title(utility);
end
epsilon=(max(meanValues)-min(meanValues))*0.05;
axis([min(meanValues)-epsilon max(meanValues)+epsilon ymin ymax]);
%legend('Linear cost','Geometric cost','FontSize',13)%,'Location','east')
xlabel('Stimuli''s magnitude')
ylabel('Reaction time')
set(gca,'FontSize',13)
pbaspect([3 2 2])
filename = strcat('simFigs/vs-scatter-full-',strrep(suffix,'.',''),'.pdf');
%saveas(gcf,filename);
end

%% Plot boundaries for varying time and magnitude
for utility = ["linear"]%, "logLm" ]%,"tan", "logHm", "sqrt"] % WARNING ** This loop works only for plotting NOT for computing
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

geo_values = [false true];
%geo_values = false;
for geo = geo_values 
if geo; geometric=true; else; geometric=false; end
if geo; geometric=true; singleDecisions=true; else; geometric=false; end
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
        try
            fprintf('loading projection...');
            load(filename, 'dbIS');
            dataLoaded = true;
            fprintf('done.\n');
        catch
            disp('Could not load the projected boundaries. Recomputing the data.');
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
        zoom=Smax/8;
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
    for i = 1:length(geo_values)
        y = 0.02+0.405*(i-1);
        for j = 1:length(timeSnaps)
            x = 0.24*(4-j)-0.05;
            %fprintf("%f\n",(i-1)*4+j);
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
    %stringlist2=['R','e','w','a','r','d','>','0'];
    for i = 1:length(stringlist2)
        annotation('textbox', [0, 0.43-(i*0.045), 0.05, 0.05], 'String', stringlist2(i), 'EdgeColor', 'none', 'FontSize', 18, 'HorizontalAlignment', 'center' );
    end
end
for i = 1:length(stringlist)
    annotation('textbox', [~magnitudePlot*0.05, 0.85-(i*0.045), 0.05, 0.05], 'String', stringlist(i), 'EdgeColor', 'none', 'FontSize', 18, 'HorizontalAlignment', 'center' );
end
%Save file
if magnitudePlot; sfx=''; else sfx='-mp'; end
outfilename = strcat('simFigs/vs-matrix',sfx,'-g',num2str(geometric),strjoin(strrep(suffix,'.',''),''),'_zoom',num2str(zoom),'.pdf');
set(gcf,'Color',[1 1 1]); set(gca,'Color',[.8 .8 .8]); set(gcf,'InvertHardCopy','off');
saveas(gcf,outfilename)
end

