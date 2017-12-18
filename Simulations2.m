%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project Simulations                                                 %%%
%%% Tyler Lesthaeghe                                                    %%%
%%% E E 573                                                             %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% This file can be downloaded from GitHub at                          %%%
%%% https://git.io/vbgBL                                                %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clearvars; close all;

%%% Configuration Options %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Simulated Area Settings
xsize = 300;            % Test Area X Length (m)
ysize = 250;            % Test Area Y Length (m)
gridsize = 1;           % Transmitter Power Grid Size

% Simulated Building
buildingon = false;     % Using Simulated Building -- Set to true/false
xmitinbldg = true;      % Require Transmitter to be in Building -- true/false
bldgx = -1;             % Building X Location (m) -- Set to <0 for random
bldgy = -1;             % Building Y Location (m) -- Set to <0 for random
bldgxsize = 25;         % Building X Length (m)
bldgysize = 25;         % Building Y Length (m)
bldgnxrms = 1;          % Number of Rooms in X Direction
bldgnyrms = 1;          % Number of Rooms in Y Direction
bldgextwall = 0.1651;   % Exterior Wall Thickness (m)
bldgintwall = 0.127;    % Interior Wall Thickness (m)

% Simulated Transmitter
xmitx = 150;            % Transmitter Location (m) -- Set to <0 for random
xmity = 125;            % Transmitter Location (m) -- Set to <0 for random
xmitpower = 28;         % Transmitter Power (mW)
xmitfreq = 2.4e9;       % Transmitter Frequency (Hz)

% Simulated Walk
walkx = -1;             % Walk X Location (m) -- Set to <0 for random
walky = -1;             % Walk Y Location (m) -- Set to <0 for random
meanwalkv = 1.4;        % Average Walk Speed (m/s)
accdt = 0.0096;         % Accelerometer Sampling Period (s) [LSM6DSL Doc]
gpsdt = 1;              % GPS Sampling Period (s)
walktime = 600;         % Walk Time (s)
arwalk = 0.99;          % Walk Distance a parameter
atwalk = 0.999;         % Walk Turn a parameter
stdrwalk = 0.01;        % Standard Deviation of Walk Distance
stdtwalk = 40*pi/180;   % Standard Deviation of Walk Turn

% Ensemble Filter Settings
J = 10;                 % Ensemble Kalman Filter Order

%%% Constants %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
s_scatter = 7;          % Gaussian StdDev for Scattering (dB) [Anderson]
att_intwall = 10;       % Attenuation in Interior Wall (dB) [Atmel]
att_extwall = 20;       % Attenuation in Interior Wall (dB) [Atmel]

g = 9.80665;            % Acceleration Due to Gravity (m/s^2) [Wikipedia]

s_acc = 2e-3*g;         % Gaussian StdDev for Accelerometer (m/s^2) [LSM6DSL Doc]
s_gps = 6.7/3;          % Gaussian StdDev for GPS (m) [Wikipedia]

s_fused = 1.7;          % Assumed Value for Fused Data StdDev (m)

%%% Simulation Code Below Here %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initialization
rng('shuffle');

% Set Up Outer Bounds
bounds = zeros(4,4);
bounds(1,1) = 0; bounds(1,2) = 0; bounds(1,3) = xsize; bounds(1,4) = 0;
bounds(2,1) = xsize; bounds(2,2) = 0; bounds(2,3) = xsize; bounds(2,4) = ysize;
bounds(3,1) = xsize; bounds(3,2) = ysize; bounds(3,3) = 0; bounds(3,4) = ysize;
bounds(4,1) = 0; bounds(4,2) = ysize; bounds(4,3) = 0; bounds(4,4) = 0;

% Compute Building Location
if buildingon
    
    % Compute random location for building
    if bldgx < 0
        bldgx = rand(1)*bldgxsize;
    end
    if bldgy < 0
        bldgy = rand(1)*bldgysize;
    end
    
    % Warn User That They Have Chosen Invalid Options
    if xmitinbldg && (xmitx >=0 || xmity >= 0)
        disp('WARNING! xmitinbldg being ignored');
    end
    
    % Set Transmitter Location Inside Building
    if xmitinbldg
        if xmitx < 0
            xmitx = bldgx + rand(1)*bldgxsize;
        end
        if xmity < 0
            xmity = bldgy + rand(1)*bldgysize;
        end        
    end
    
    % Set Up Wall Segment Vector
    walls = zeros(4+bldgnxrms+bldgnxrms-2,7); % 2nd Parameter - x0,x1,y0,y1,xory,thickness,att
    
    % Top Exterior Wall
    walls(1,1) = bldgx + (bldgextwall/2);               % x0
    walls(1,2) = bldgy + (bldgextwall/2);               % x1
    walls(1,3) = bldgx + bldgxsize - (bldgextwall/2);   % y0
    walls(1,4) = bldgy + (bldgextwall/2);               % y1
    walls(1,5) = 0;                                     % xory
    walls(1,6) = bldgextwall;                           % thickness
    walls(1,7) = att_extwall;                           % att
    
    % Bottom Exterior Wall
    walls(2,1) = bldgx + (bldgextwall/2);               % x0
    walls(2,2) = bldgy + bldgysize - (bldgextwall/2);   % x1
    walls(2,3) = bldgx + bldgxsize - (bldgextwall/2);   % y0
    walls(2,4) = bldgy + bldgysize - (bldgextwall/2);   % y1
    walls(2,5) = 0;                                     % xory
    walls(2,6) = bldgextwall;                           % thickness
    walls(2,7) = att_extwall;                           % att
    
    % Left Exterior Wall
    walls(3,1) = bldgx + (bldgextwall/2);               % x0
    walls(3,2) = bldgy + (bldgextwall/2);               % x1
    walls(3,3) = bldgx + (bldgextwall/2);               % y0
    walls(3,4) = bldgy + bldgysize - (bldgextwall/2);   % y1
    walls(3,5) = 1;                                     % xory
    walls(3,6) = bldgextwall;                           % thickness
    walls(3,7) = att_extwall;                           % att
    
    % Right Exterior Wall
    walls(4,1) = bldgx + bldgxsize - (bldgextwall/2);   % x0
    walls(4,2) = bldgy + (bldgextwall/2);               % x1
    walls(4,3) = bldgx + bldgxsize - (bldgextwall/2);   % y0
    walls(4,4) = bldgy + bldgysize - (bldgextwall/2);   % y1
    walls(4,5) = 1;                                     % xory
    walls(4,6) = bldgextwall;                           % thickness
    walls(4,7) = att_extwall;                           % att
    
    for i = 1:bldgnxrms-1
        % Vertical Interior Wall
        j = i+4;
        walls(j,1) = bldgx + bldgxsize*i/bldgnxrms;         % x0
        walls(j,2) = bldgy + (bldgextwall/2);               % x1
        walls(j,3) = bldgx + bldgxsize*i/bldgnxrms;         % y0
        walls(j,4) = bldgy + bldgysize - (bldgextwall/2);   % y1
        walls(j,5) = 1;                                     % xory
        walls(j,6) = bldgintwall;                           % thickness
        walls(j,7) = att_intwall;                           % att
    end
    for i = 1:bldgnyrms-1
        % Horizontal Interior Wall
        j = i+4+bldgnxrms-1;
        walls(j,1) = bldgx + (bldgextwall/2);               % x0
        walls(j,2) = bldgy + bldgysize*i/bldgnyrms;         % x1
        walls(j,3) = bldgx + bldgxsize - (bldgextwall/2);   % y0
        walls(j,4) = bldgy + bldgysize*i/bldgnyrms;         % y1
        walls(j,5) = 0;                                     % xory
        walls(j,6) = bldgintwall;                           % thickness
        walls(j,7) = att_intwall;                           % att
    end

else
    bldgxsize = 0; %#ok<UNRCH>
    bldgysize = 0;   
end

% Determine Simulated Transmitter Location
if xmitx < 0
    xmitx = rand(1)*xsize;
end
if xmity < 0
    xmity = rand(1)*ysize;
end

% Set Up Look Up Table
ngridx = round(xsize/gridsize);
ngridy = round(ysize/gridsize);
gridx = linspace(0, xsize, ngridx);
gridy = linspace(0, ysize, ngridy);
powergrid = zeros(ngridx, ngridy);

% Transmitter Parameters
xmitpower_dBm = 10*log10(xmitpower);
lambda = 3e8/xmitfreq;

% Compute WiFi Signal Strength Everywhere For Later Lookup
for x=1:ngridx
    for y=1:ngridy
        d = sqrt( (gridx(x)-xmitx)^2 + (gridy(y)- xmity)^2 );
        powergrid(x,y) = xmitpower_dBm - 20*log10(4*pi*d/lambda) - normrnd(0,s_scatter);
        if buildingon
           for i=1:size(walls,1)
               if Intersection(xmitx, xmity, gridx(x), gridy(y), walls(i,1), walls(i,2), walls(i,3), walls(i,4))
                   powergrid(x,y) = powergrid(x,y) - walls(i,7);
               end
           end
        end
    end
end

powergridlookup = @(x,y) interp2(gridx, gridy, powergrid', x, y, 'spline');

% Plot Power
figure(1); hold on;
[XX,YY] = meshgrid(gridx, gridy);
surf(XX', YY', powergrid, 'EdgeColor', 'None');
view(2);
xlabel('Position, $x$ (m)', 'Interpreter', 'latex');
ylabel('Position, $y$ (m)', 'Interpreter', 'latex');
cb = colorbar(); %caxis([-80, -40]);
ylabel(cb, 'Received Power, $P$ (dBm)', 'Interpreter', 'latex');
axis equal; ta = daspect(); daspect(ta); % Force Axis To Remain Equal
pbaspect([xsize ysize xsize*ysize]);

% Plot Transmitter
plot3(xmitx, xmity, 100, 'ko');

% Plot Building
if buildingon
    for i=1:size(walls,1)
        if walls(i,5)==0
            patch([walls(i,1)-(walls(i,6)/2) walls(i,3)+(walls(i,6)/2) walls(i,3)+(walls(i,6)/2) walls(i,1)-(walls(i,6)/2)], [walls(i,2)-(walls(i,6)/2) walls(i,2)-(walls(i,6)/2) walls(i,4)+(walls(i,6)/2) walls(i,4)+(walls(i,6)/2)], [100,100,100,100], 'k', 'EdgeColor', 'None')
        else
            patch([walls(i,1)-(walls(i,6)/2) walls(i,1)+(walls(i,6)/2) walls(i,3)+(walls(i,6)/2) walls(i,3)-(walls(i,6)/2)], [walls(i,2)-(walls(i,6)/2) walls(i,2)-(walls(i,6)/2) walls(i,4)+(walls(i,6)/2) walls(i,4)+(walls(i,6)/2)], [100,100,100,100], 'k', 'EdgeColor', 'None')
        end
    end
end

% Find Random Starting Place
if walkx < 0
    walkx = rand(1)*(xsize-bldgxsize);
    if buildingon && walkx > bldgx
        walkx = walkx + bldgxsize;
    end
end
if walky < 0
    walky = rand(1)*(ysize-bldgysize);
    if buildingon && walky > bldgy
        walky = walky + bldgysize;
    end
end

% Plot Walk Start Location
plot3(walkx, walky, 100, 'ko');

% Set Up Time Steps
dispdt = min([accdt, gpsdt])*100;
dispt = 0:dispdt:walktime;
gpst = 0:gpsdt:walktime;
dispN = size(dispt,2);

% Compute Semi-Random Walk
urwalk = normrnd(0, stdrwalk*sqrt(1-arwalk^2), 1, dispN);
walkx = [walkx, walkx, zeros(1,dispN-2)];
walky = [walky, walky, zeros(1,dispN-2)];
walkrerr = zeros(1,dispN); walkrerr(1) = 0;
walkterr = zeros(1,dispN); walkterr(1) = 0;
walkr = zeros(1, dispN); walkr(1) = meanwalkv*dispdt;
walkt = zeros(1, dispN); walkt(1) = 0;
if buildingon
    for i = 3:dispN
        bounddist = zeros(4,1);
        for j = 1:4
            bounddist(j) = Distance(bounds(j,1), bounds(j,2), bounds(j,3), bounds(j,4), walkx(i-1), walky(i-1));
        end
        [~, bids] = sort(bounddist);
        bid(i) = bids(1);
        [bx, by] = NearestPoint(bounds(bid(i),1), bounds(bid(i),2), bounds(bid(i),3), bounds(bid(i),4), walkx(i-1), walky(i-1));
        bth(i) = atan2(by-walky(i-1),bx-walkx(i-1));
        if bth(i) < 0
            bth(i) = bth(i) + 2*pi;
        end
        bdist = bounddist(bids(1));
              
        walldist = zeros(4,1);
        for j = 1:4
            walldist(j) = Distance(walls(j,1), walls(j,2), walls(j,3), walls(j,4), walkx(i-1), walky(i-1));
        end
        [~, wids] = sort(walldist);
        wid(i) = wids(1);
        [wx, wy] = NearestPoint(walls(wid(i),1), walls(wid(i),2), walls(wid(i),3), walls(wid(i),4), walkx(i-1), walky(i-1));
        wth(i) = atan2(wy-walky(i-1),wx-walkx(i-1));
        if wth(i) < 0
            wth(i) = wth(i) + 2*pi;
        end
        wdist = walldist(wids(1));
       
        if (bth(i) < wth(i))
            wthbth(i) = bth(i) + (wdist/(wdist+bdist))*abs(bth(i) - wth(i));
        else
            wthbth(i) = wth(i) - (bdist/(wdist+bdist))*abs(wth(i) - bth(i));
        end
        
        utwalk(i) = normrnd(0, stdtwalk*sqrt(1-atwalk^2), 1);
        walkrerr(i) = arwalk * walkrerr(i-1) + urwalk(i);
        walkr(i) = (meanwalkv + walkrerr(i))*dispdt;
        walkterr(i) = atwalk * walkterr(i-1) + utwalk(i);
        walkt(i) = wthbth(i) + walkterr(i);
        walkx(i) = walkx(i-1) + walkr(i) * cos(walkt(i));
        walky(i) = walky(i-1) + walkr(i) * sin(walkt(i));        
    end
else
    for i = 3:dispN
        bounddist = zeros(4,1);
        for j = 1:4
            bounddist(j) = Distance(bounds(j,1), bounds(j,2), bounds(j,3), bounds(j,4), walkx(i-1), walky(i-1));
        end
        [~, bids] = sort(bounddist);
        bid(i) = bids(1);
        [bx, by] = NearestPoint(bounds(bid(i),1), bounds(bid(i),2), bounds(bid(i),3), bounds(bid(i),4), walkx(i-1), walky(i-1));
        bth(i) = atan2(by-walky(i-1),bx-walkx(i-1));
        if bth(i) < 0
            bth(i) = bth(i) + 2*pi;
        end
        bdist = bounddist(bids(1));
             
        wth(i) = atan2(xmity-walky(i-1),xmitx-walkx(i-1));
        if wth(i) < 0
            wth(i) = wth(i) + 2*pi;
        end
        wdist = sqrt((xmitx-walky(i-1))^2 + (xmity-walkx(i-1))^2);
       
        if (bth(i) < wth(i))
            wthbth(i) = bth(i) - (wdist/(wdist+bdist))*abs(bth(i) - wth(i));
        else
            wthbth(i) = wth(i) + (bdist/(wdist+bdist))*abs(wth(i) - bth(i));
        end
        
        utwalk(i) = normrnd(0, stdtwalk*sqrt(1-atwalk^2), 1);
        walkrerr(i) = arwalk * walkrerr(i-1) + urwalk(i);
        walkr(i) = (meanwalkv + walkrerr(i))*dispdt;
        walkterr(i) = atwalk * walkterr(i-1) + utwalk(i);
        walkt(i) = wthbth(i) + walkterr(i);
        walkx(i) = walkx(i-1) + walkr(i) * cos(walkt(i));
        walky(i) = walky(i-1) + walkr(i) * sin(walkt(i));        
    end
end

% Plot Real Walk on Map
plot3(walkx, walky, zeros(size(walkx)) + 100, 'r-');

% Compute Real Velocity and Acceleration
walkvx = diff(walkx)/dispdt; walkvy = diff(walky)/dispdt;
walkax = diff(walkvx)/dispdt; walkay = diff(walkvy)/dispdt;

% Plot Real Walk on Plot
figure(2); hold on;
ax1 = subplot(3,2,1); plot(dispt, walkx); 
ylabel('Position (m)', 'Interpreter', 'latex');
title('$x$ Component', 'Interpreter', 'latex');
ax2 = subplot(3,2,2); plot(dispt, walky);
title('$y$ Component', 'Interpreter', 'latex');
ax3 = subplot(3,2,3); plot(dispt(1:end-1), walkvx); 
ylabel('Velocity (m/s)', 'Interpreter', 'latex');
ax4 = subplot(3,2,4); plot(dispt(1:end-1), walkvy);
ax5 = subplot(3,2,5); plot(dispt(1:end-2), walkax); 
xlabel('Time, $t$ (s)', 'Interpreter', 'latex');
ylabel('Acceleration (m/s$^2$)', 'Interpreter', 'latex');
ax6 = subplot(3,2,6); plot(dispt(1:end-2), walkay); 
xlabel('Time, $t$ (s)', 'Interpreter', 'latex');
linkaxes([ax1,ax2,ax3,ax4,ax5,ax6],'x');



% Compute Simulated Accelerometer Data Capture
% This assumes we've already done the work necessary to get accelerometer
% data transformed into the appropriate coordinate axis -- our goal is to
% simply capture the error from the accelerometer

% Sample the real acceleration data
acct = 0:accdt:walktime;
accax_base = interp1(dispt(1:end-2), walkax, acct, 'pchip');
accay_base = interp1(dispt(1:end-2), walkay, acct, 'pchip');

% Compute accelerometer data (real positions plus noise)
accax = accax_base + normrnd(0, s_acc, 1, length(accax_base));
accay = accay_base + normrnd(0, s_acc, 1, length(accay_base));

% Sample the real position data
gpst = 0:gpsdt:walktime; 
gpsx_base = interp1(dispt(1:end), walkx, gpst, 'pchip');
gpsy_base = interp1(dispt(1:end), walky, gpst, 'pchip');

% Compute GPS data (real positions plus noise)
gpsx = gpsx_base + normrnd(0, s_gps, 1, length(gpsx_base));
gpsy = gpsy_base + normrnd(0, s_gps, 1, length(gpsy_base));

% Run Kalman Filter to Fuse GPS and Accelerometer Data
% Doing this "Real Time" to simulate real world scenario

% Constant Matrices
A = [1, gpsdt, 0, 0; 0, 1, 0, 0; 0, 0, 1, gpsdt; 0, 0, 0, 1];
Q = [s_acc^2, 0, 0, 0; 0, s_acc^2, 0, 0; 0, 0, s_acc^2, 0; 0, 0, 0, s_acc^2];
R = [s_gps^2, 0, 0, 0; 0, s_gps^2, 0, 0; 0, 0, s_gps^2, 0; 0, 0, 0, s_gps^2];
% B will need to be handled in real time since we will need to integrate
% existing data between our previous and current state update
H = [1, 0, 0, 0; 0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1];

% Initial State Prediction
X = [0; 0; 0; 0];  % Let's assume zeros for now -- should update quickly
P = [0, 0, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0];

% Save Position Parameters
fusex = zeros(1,length(gpst));
fusey = zeros(1,length(gpst));
fusevx = zeros(1,length(gpst));
fusevy = zeros(1,length(gpst));

% Begin Kalman Filter Loop
i=1;
for t = gpst
    
    if i == 1
        curaccax = accax(acct<=t);
        curaccay = accay(acct<=t);
    else
        curaccax = accax(acct>gpst(i-1) & acct<=t);
        curaccay = accay(acct>gpst(i-1) & acct<=t);
    end
    curvx = sum(curaccax)*accdt;
    curvy = sum(curaccay)*accdt;
    curx = sum(curvx)*accdt;
    cury = sum(curvy)*accdt;
    
    Xp = A * X + [curx; curvx; cury; curvy];
    Pp = A * P * A' + Q;
    
    if i == 1
        Y = [0; 0; 0; 0] - H * Xp;
    else
        Y = [gpsx(i); 
            (gpsx(i)-gpsx(i-1))*gpsdt; 
            gpsy(i); 
            (gpsy(i)-gpsy(i-1))*gpsdt] - H * Xp;
    end
    
    S = H*Pp*H' + R;
    K = Pp*H'*S^(-1);
    X = Xp + K*Y;
    P = ([1,0,0,0;0,1,0,0;0,0,1,0;0,0,0,1] - K*H)*Pp;   
    
    fusex(i) = X(1);
    fusey(i) = X(3);
    fusevx(i) = X(2);
    fusevy(i) = X(4);
    
    i=i+1;
end

% Plot on Map
figure(1); hold on;
plot(fusex, fusey, 'gx');

% Plot on Real Data Plot for Comparison
figure(2); hold on;
subplot(3,2,1); hold on; plot(gpst, fusex, 'gx', 'MarkerSize', 0.5);
subplot(3,2,2); hold on; plot(gpst, fusey, 'gx', 'MarkerSize', 0.5);
subplot(3,2,3); hold on; plot(gpst, fusevx, 'gx', 'MarkerSize', 0.5);
subplot(3,2,4); hold on; plot(gpst, fusevy, 'gx', 'MarkerSize', 0.5);


fusexerr = fusex - gpsx_base;
fuseyerr = fusey - gpsy_base;



% Run Actual Kalman Filter to Track Radio Position

% Constant Matrices
H = eye(3);
G = [s_fused^2, 0, 0; 0, s_fused^2, 0; 0, 0, s_scatter^2];
BBPower = @(x,y,p,a,b) p - 20.*log10(4.*pi.*sqrt((x-a).^2+(y-b).^2)./lambda);

% Prepare Initial Ensemble
U = zeros(J,3);
h = 1;
for i=linspace(0,xsize,J)
    for j=linspace(0,ysize,J)
        for k=linspace(0,20,J)
            U(h,:) = [i,j,k];
            h=h+1;
        end
    end
end

% Lookup Table for Actual X and Y Positions
actualx=@(t)interp1(dispt,walkx,t);
actualy=@(t)interp1(dispt,walky,t);

% Store Mean and Standard Deviation of Ensemble Over Time
outmeans = zeros(i,3);
outstds = zeros(i,3);

% Perform the actual EnKF Loop
i=1;
figure(1); hold on;
for t = gpst(1:end)
    
    % Safety to prevent NaNs from showing up when GPS goes out of range
    if fusex(i)<0  || fusex(i)>xsize || fusey(i)<0 || fusey(i)>ysize
        outmeans(i,:) = [NaN, NaN, NaN];
        outstds(i,:) = [NaN, NaN, NaN];
        i=i+1;
        continue
    end    
    
    % Compute Means
    UBar = mean(U);
    W(:,1) = fusex(i)+normrnd(0,s_fused,J*J*J,1);
    W(:,2) = fusey(i)+normrnd(0,s_fused,J*J*J,1);
    W(:,3) = BBPower(U(:,1), U(:,2), xmitpower_dBm, W(:,1), W(:,2))+normrnd(0,s_scatter,J*J*J,1);
    WBar = mean(W);
    
    % Compute Covariances
    EW = W-WBar;
    EU = U-UBar;
    Cww = EW'*EW/J/J/J;
    Cuw = EU'*EW/J/J/J;
    
    % Get Current Measurement
    meas = [fusex(i), fusey(i), powergridlookup(actualx(i),actualy(i))];
    
    % Handle Bad Data
    if any(isnan(meas))
        outmeans(i,:) = [NaN, NaN, NaN];
        outstds(i,:) = [NaN, NaN, NaN];
        i=i+1;
        continue
    end
    
    % Update Ensemble
    for j=1:J*J*J
        U(j,:) = U(j,:)'+Cuw*(Cww+G)^(-1)*(meas-W(j,:))';
    end
    
    % Save Current Ensemble State
    outmeans(i,:) = mean(U);
    outstds(i,:) = std(U);
    
    % Plot for live tracking on map
    figure(1); hold on;
    plot3(outmeans(i,1),outmeans(i,2),100,'cx');
    drawnow()
    
    % Plot live histograms
    figure(4);
    subplot(1,2,1);
    histogram(U(:,1));
    xlabel('Position, x (m)', 'Interpreter', 'latex');
    xlim([0,xsize]);
    subplot(1,2,2); cla;
    histogram(U(:,2));
    xlabel('Position, y (m)', 'Interpreter', 'latex');
    xlim([0,ysize]);
    
    i=i+1;
end

% Plot Final EnKF Results
figure(3); 
ax7=subplot(3,2,1); hold on;
plot(gpst(1:length(outmeans)), outmeans(:,1));
plot([0,gpst(length(outmeans))],[xmitx xmitx],'k--');
title('$x$ Component', 'Interpreter', 'latex');
ylabel('Position (m)', 'Interpreter', 'latex');
ax8=subplot(3,2,2); hold on;
plot(gpst(1:length(outmeans)), outmeans(:,2));
plot([0,gpst(length(outmeans))],[xmity xmity],'k--');
title('$y$ Component', 'Interpreter', 'latex');
ax9=subplot(3,2,3); hold on;
plot(gpst(1:length(outstds)), outstds(:,1));
ylabel('Std. Deviation (m)', 'Interpreter', 'latex');
ax10=subplot(3,2,4); hold on;
plot(gpst(1:length(outstds)), outstds(:,2));
ax11=subplot(3,2,5); hold on;
plot(gpst(1:length(outmeans)), outmeans(:,1)-xmitx);
xlabel('Time, $t$ (s)', 'Interpreter', 'latex');
ylabel('Error (m)', 'Interpreter', 'latex');
ax12=subplot(3,2,6); hold on;
plot(gpst(1:length(outmeans)), outmeans(:,1)-xmity);
xlabel('Time, $t$ (s)', 'Interpreter', 'latex');
linkaxes([ax7,ax8,ax9,ax10],'x');

% Let's Try an Extended Kalman Filter
i=3;
figure(1); hold on;
ekf_Q = eye(3)*s_fused^2; ekf_Q(3,3) = s_scatter^2;
ekf_R = s_scatter^2;
ekf_X = zeros(3,length(gpst(3:end)));
ekf_Pvals = zeros(3,length(gpst(3:end)));
ekf_x = [0; 0; 10];
ekf_P = eye(3);
for t = gpst(3:end)
    cur_f = @(x) ekf_f(x, fusex(i-2), fusey(i-2), powergridlookup(actualx(i-2), actualy(i-2)), fusex(i-1), fusey(i-1), powergridlookup(actualx(i-1), actualy(i-2)), fusex(i), fusey(i), lambda, s_scatter);
    cur_h = @(x) ekf_h(x, fusex(i), fusey(i), lambda, xmitpower_dBm, s_scatter);
    cur_z = powergridlookup(actualx(i), actualy(i));
    
    xp = cur_f(ekf_x);
    dxp = Jacobian(cur_f, ekf_x);
    ekf_P = dxp*ekf_P*dxp'+ekf_Q;
    z = cur_h(xp);
    dH = Jacobian(cur_h, xp);
    Pph = ekf_P*dH';
    K = Pph*inv(dH*Pph+ekf_R);
    ekf_x = xp + K*(cur_z - z);
    ekf_P = ekf_P - K*Pph';

    ekf_X(:,i-2) = ekf_x;
    ekf_Pvals(:,i-2) = [ekf_P(1,1);ekf_P(2,2); ekf_P(3,3)];
    figure(1); hold on;
    plot(ekf_x(1),ekf_x(2),'mx');
    drawnow();
    i=i+1;
end


% Plot Final EKF Results
figure(5); 
ax11=subplot(2,3,1); hold on;
plot(gpst(3:size(ekf_X,2)+2), ekf_X(1,:)');
plot([0,gpst(size(ekf_X,2)+2)],[xmitx xmitx],'k--');
title('$x$ Component', 'Interpreter', 'latex');
ylabel('Position (m)', 'Interpreter', 'latex');
ax12=subplot(2,3,2); hold on;
plot(gpst(3:size(ekf_X,2)+2), ekf_X(2,:)');
plot([0,gpst(size(ekf_X,2)+2)],[xmity xmity],'k--');
title('$y$ Component', 'Interpreter', 'latex');
ax13=subplot(2,3,3); hold on;
plot(gpst(7:size(ekf_X,2)+2), ekf_X(3,5:end));
plot([0,gpst(length(outmeans))],[xmitpower_dBm xmitpower_dBm],'k--');
title('Power', 'Interpreter', 'latex');
ylabel('Power (dBm)', 'Interpreter', 'latex');
ax14=subplot(2,3,4); hold on;
plot(gpst(3:length(ekf_Pvals)+2), ekf_X(1,:)'-xmitx);
xlabel('Time, $t$ (s)', 'Interpreter', 'latex');
ylabel('Error (m)', 'Interpreter', 'latex');
ax15=subplot(2,3,5); hold on;
plot(gpst(3:length(ekf_Pvals)+2), ekf_X(2,:)'-xmity);
xlabel('Time, $t$ (s)', 'Interpreter', 'latex');
ax16=subplot(2,3,6); hold on;
plot(gpst(7:length(ekf_Pvals)+2), ekf_X(3,5:end)'-xmitpower_dBm);
ylabel('Error (dBm)', 'Interpreter', 'latex');
xlabel('Time, $t$ (s)', 'Interpreter', 'latex');
linkaxes([ax11,ax12,ax13,ax14,ax15,ax16],'x');
xlim([0, gpst(end)]);

%%% Utility Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function xnew = ekf_f(x, x1, y1, Rx1, x2, y2, Rx2, x3, y3, lambda, s_scatter)
% EKF_F Computes the estimated state based on the current state and
% previous measurements of received power at different locations
%   xnew = ekf_f(x, x1, y1, Rx1, x2, y2, Rx2, x3, y3, lambda)
%       x = State Vector
%       x1, y1 = Previous Measurement Location 1
%       Rx1 = Signal Level at Location 1
%       x2, y2 = Previous Measurement Locaiton 2
%       Rx2 = Signal Level at Location 2
%       x3, y3 = New Location
%       lambda = Wavelength for Normalization in Friis Equation

    % Current State Values for Estimated Transmitter Location and Power
    px = x(1);  py = x(2);  tx = x(3);
    
    % Predict radius for next state using existing estimate
    r3 = sqrt((x3-px)^2 + (y3-py)^2);
    
    % Compute radius for previous two measurements
    r1 = sqrt((x1-px)^2 + (y1-py)^2);
    r2 = sqrt((x2-px)^2 + (y2-py)^2);
    
    % Set up system of equations for solution
    A = [x2-x1, y2-y1; x3-x1, y3-y1];
    b = 0.5 * [(r1^2-r2^2)+(x2^2+y2^2)-(x1^2+y1^2); (r1^2-r3^2)+(x3^2+y3^2)-(x1^2+y1^2)];
    
    % Compute next transmit power estimate as an average of the last two 
    % Probably reasonable given small step size
    Tx1 = Rx1 + 20*log10(4*pi*r1/lambda) - normrnd(0, s_scatter);
    Tx2 = Rx2 + 20*log10(4*pi*r2/lambda) - normrnd(0, s_scatter);
    
    % Compute and return results    
    xnew = [A\b;mean([Tx1,Tx2])];
end

function z = ekf_h(x, x3, y3, lambda, xmitpower, s_scatter)
% EKF_H Compute Expected Measurement Result Based on Current State Estimate
%   z = ekf_h(x, x3, y3, lambda, xmitpower)
%       x = Current State Vector
%       x3, y3 = Current Measurement Location
%       lambda = Wavelength Normalization Factor for Friis Equation
%       xmitpower = Correct Constant Transmitter Power in dBm

    px = x(1);  py = x(2);
    r3 = sqrt((x3-px)^2 + (y3-py)^2);
    z = xmitpower - 20*log10(4*pi*r3/lambda) + normrnd(0, s_scatter);
end

function df = Jacobian(f,x)
% JACOBIAN Compute Jacobian of f at x using complex step differentiation
%   df = Jacobian(f,x)
    y = f(x);
    m = numel(x); n = numel(y);
    df = zeros(n,m);
    h = eps;
    for j = 1:m
        z = x;
        z(j) = h*1i + z(j);
        df(:,j) = imag(f(z))/h;
    end
end

function intersects = Intersection(x0, x1, y0, y1, a0, a1, b0, b1)
% INTERSECTION Calculate Intersection of Two Line Segments
%   intersects = Intersection(x0, x1, y0, y1, a0, a1, b0, b1) calculates
%     the intersection of the line segment (x0,x1) <-> (y0,y1) and the
%     segment (a0,a1) <-> (b0,b1), returning true if intersecting, and
%     false if not intersecting
%
% From Numerical Recipes, 3rd Ed., Press, et al.
    s = ((x0-y0)*(a1-x1) - (x1-y1)*(a0-x0)) / ((b0-a0)*(x1-y1) - (b1-a1)*(x0-y0));
    t = ((a0-x0)*(b1-a1) - (a1-x1)*(b0-a0)) / ((b0-a0)*(x1-y1) - (b1-a1)*(x0-y0));

    if s>0 && s<1 && t>0 && t<1
        intersects = true;
    else
        intersects = false;
    end
end

function d = Distance(x0, x1, y0, y1, a0, a1)
% DISTANCE Caluclate distance from point A to Line Segment X-Y
%   d = Distance(x0, x1, y0, y1, a0, a1) calculates the minimum distance
%       between the line segment (x0,x1) <-> (y0,y1) and the point (a0,a1)

    vx = x0-a0;    vy = x1 - a1;
    ux = y0 - x0;  uy = y1 - x1;
    lensq = ux^2 + uy^2;
    detp = -vx*ux + -vy*uy;
    
    if detp < 0
        d = norm([x0,x1]-[a0,a1],2);
    elseif detp > lensq
        d = norm([y0,y1]-[a0,a1],2);
    else
        d = abs(ux*vy-uy*vx)/sqrt(lensq);
    end
end

function [x,y] = NearestPoint(x0, x1, y0, y1, a0, a1)
% NEARESTPOINT Calculate the nearest point on line segment X-Y from point A

    vx = x0 - y0;  vy = x1 - y1;
    ux = x0 - a0;  uy = x1 - a1;
    
    t = (vx*ux + vy*uy)/(vx*vx + vy*vy);
    
    if t > 0 && t < 1
        x = (1-t)*x0 + t*y0;
        y = (1-t)*x1 + t*y1;
    else
        if sqrt((x0-a0)^2 + (x1-a1)^2) < sqrt((y0-a0)^2 + (y1-a1)^2)
            x = x0;
            y = x1;
        else
            x = y0;
            y = y1;
        end
    end

end
