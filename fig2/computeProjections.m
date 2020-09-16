function computeProjections(utility)
addpath('../shared')
global gamm geometric epsil pie; % (JARM 23rd August '19)
global valscale; % (JARM 7th October '19)
geometric = true; % (JARM 23rd August '19) use geometric discounting for future rewards
fprintf("Script started with utlity %s and geometric %d\n",utility, geometric)
gamm = 0.6; % (JARM 23rd August '19) geometric discount factor for future rewards
epsil = 0; % (JARM 11th September '19) epsilon error to add to co-planar services to compute convex hull (required to check geometric discounting results; deprecated)
pie = 0; % (JARM 27th May '20) input-dependent noise scaling 
valscale = 0.5; % (JARM 7th October '19) move triangle along diagonal as option values scale)
maxval = 4; % (JARM 6th March '20) maximum utility for logistic utility function
%utility='tan';
tic;
Smax = 8;      % Grid range of states space (now we assume: S = [(Rhat1+Rhat2)/2, (Rhat1-Rhat2)/2]); Rhat(t) = (varR*X(t)+varX)/(t*varR+varX) )
resSL  = 15;      % Grid resolution of state space
resS = 151;      % Grid resolution of state space
tmax = 37;       % Time limit
dt   = .0625;       % Time step
c    = 0.1;       % Cost of evidence accumulation
tNull = .25;     % Non-decision time + inter trial interval
g{1}.meanR = 0; % Prior mean of state (dimension 1)
g{1}.varR  = 5; % Prior variance of stte
g{1}.varX  = 2 + (pie * valscale); % Observation noise variance
g{2}.meanR = 0; % Prior mean of state (dimension 2)
g{2}.varR  = 5; % Prior variance of state
g{2}.varX  = 2 + (pie * valscale); % Observation noise variance
g{3}.meanR = 0; % Prior mean of state (dimension 3)
g{3}.varR  = 5; % Prior variance of state
g{3}.varX  = 2 + (pie * valscale); % Observation noise variance
t = 0:dt:tmax;
Slabel = {'r_1^{hat}', 'r_2^{hat}', 'r_3^{hat}'};
myCol = [1 0 0; 0 1 0; 0 0 1];

if contains(utility , 'all')
    utilities=["linear", "tan", "logLm", "logHm", "sqrt"];
else
    utilities=utility;
end
%for utility = ["tan", "logLm", "logHm"]
for utility = utilities 
    fprintf("Loop started with utlity %s and geometric %d\n",utility, geometric)
%     computeProjectionsFiles(utility);
%     clear dbIS V0 V D EVnext rho Ptrans iStrans;
% end

% Select utililty function:
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

% Plot boundaries for varying time and magnitude
computeDecisionBoundaries=false; % if true the code will compute the decision boundaries, if false will try to load the decision boundaries from the directory rawData (if it fails, it will recompute the data)
singleDecisions=true; % if true we model single decision (i.e. expected future rewards=0); if false we compute the expected future reward (rho_)
meanValues=-0.5:1:2.5; % mean reward values to be tested
timeSnaps=0:0.333:1; % mean reward values to be tested
timeSnaps=0:0.25:0.75;
if contains( ['tan','logHm'],utility )
    %timeSnaps=0:0.1:0.3; % mean reward values to be tested are different
    timeSnaps=0:0.25:0.75; % mean reward values to be tested are different
elseif contains( 'logLm',utility )
    timeSnaps=0:0.25:0.75; % mean reward values to be tested are different
end
%meanValues=-0.5:0.5:0.5;
%meanValues=1.5;
%timeSnaps=0.3;
j=1; % subplot counter

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
for meanValue = meanValues
    fixedMeanValue = 1.5;
    g{1}.meanR = fixedMeanValue; g{2}.meanR = fixedMeanValue; g{3}.meanR = fixedMeanValue; % fix prior of mean to a fixed value
    valscale = (3*meanValue + Smax)/3; % moving to the correct 2d projection plane
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
            if isfile(filename)
                fprintf('loading projection...');
                dataLoaded = true;
                fprintf('done.\n');
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
                iS0 = [findnearest(g{1}.meanR, Sscale) findnearest(g{2}.meanR, Sscale) findnearest(g{3}.meanR, Sscale)]; % this line is not really needed
                if singleDecisions
                    rho_ = 0; % we assume single decisions
                else
                    SscaleL  = linspace(-Smax, Smax, resSL);
                    iS0 = [findnearest(g{1}.meanR, SscaleL) findnearest(g{2}.meanR, SscaleL) findnearest(g{3}.meanR, SscaleL)]; % this line is not really needed
                    [SLow{1},SLow{2},SLow{3}] = ndgrid(SscaleL, SscaleL, SscaleL);
                    for iC = 3:-1:1;  RhLow{iC} = utilityFunc(SLow{iC});  end  % Expected reward for option iC
                    rho_ = fzero(@(rho) backwardInduction(rho,c,tNull,g,RhLow,SLow,t,dt,iS0), g{1}.meanR); % compute future rewards decisions
                    fprintf('rho for mean %d is %d...',g{1}.meanR, rho_);
                end
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
            clear dbIS V0 V EVnext rho Ptrans iStrans;
            fprintf('done.\n');
        end
    end
end
end
clear D;
end

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
end

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
end

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
end

function v = varTrans(varRh, varR, varX, t, dt)
% v = (varR * (varX + varRh)) / ((1 + t/dt) * varR + varX / dt);
v = (varR ./ (varR*(t+dt) + varX)).^2 .* (varX + varRh * dt) * dt;
end

function prob = normal3(x, m, C)
d1 = x{1} - m(1);
d2 = x{2} - m(2);
d3 = x{3} - m(3);
H = -1/2*(C\eye(3)); prob = exp(d1.*d1*H(1,1) + d1.*d2*H(1,2) + d1.*d3*H(1,3) + ...
                                d2.*d1*H(2,1) + d2.*d2*H(2,2) + d2.*d3*H(2,3) + ...
                                d3.*d1*H(3,1) + d3.*d2*H(3,2) + d3.*d3*H(3,3));
% prob = exp(-(d1.^2/C(1,1)/2 + d2.^2/C(2,2))/2);
prob = prob ./ sum(prob(:));
end

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
end

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
end

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
end

