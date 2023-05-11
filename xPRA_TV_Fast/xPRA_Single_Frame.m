%% Description
%{
Description: xRPI-LM code fast
%}
clear; close all; clc; 

% load parameters from the simulation code
load ('Ni.mat','Ni')                % Number of sensor nodes = 20 (can change from forw code)
load ('TxRxpairs.mat','TxRxpairs')  % All possible pairs of Tx and Rx nodes (total 20(20-1)=380 pairs)
Nodes=TxRxpairs; clear TxRxpairs
load ('TxRx1.mat','TxRx1')          % Location of all 20 sensors
load ('size_DOI.mat','size_DOI')    % Size of DOI = 3x3 m^2 (can change from forw code)
clear opts

% Other Physical constants
imp = 120*pi; c = 3e8; freq = 2.4e9; lambda = c/freq; k0 = 2*pi/lambda;  
TxRx = TxRx1; 
RoomLength = size_DOI; 
RoomWidth = size_DOI;

%% Important Global Paramters (can be tuned to improve the recontructions)
gridXres = 0.05;            % grid/pixel/voxel resolution in x direction (in meters)
gridYres = gridXres;        % grid or pixel resolution in y direction (in meters)
Data_type = 'Expr';         % Load data from Simulation or Experiments (Sim or Expr)
opts.mu = 24;               % Change this parameter if reconstruction is not good
opts.beta = 16;              
opts.TVnorm = 1; 
opts.maxit =5000;
opts.tol = 1E-10;
opts.nonneg = true;
%% Create 2D Mesh of all the grids in DOI
lambX=1; lambY=1;
p=RoomWidth/gridYres;   % number of grids in DOI in x direction
q=RoomLength/gridXres;  % number of grids in DOI in y direction
tx = [gridXres/2:gridXres:RoomLength-gridXres/2]-RoomLength/2;%0:gridXres:2;%1:gridYres:2;%
ty = [RoomWidth-gridYres/2:-gridYres:gridYres/2]-RoomWidth/2;
[x,y] = meshgrid(tx,ty);
M =p;

%% Area Constants                                                                    % Correction factor                                                     % Free Space permittivity
% Gridarea = gridXres*gridYres;  Gridvol = Gridarea*gridYres;                                              % Area of one grid
cellrad = (sqrt(gridXres^2/pi)*2)/2; % radius of cells
Gridarea=(4*pi*cellrad/(2*k0))*besselj(1,k0*cellrad);

%% Definition of the Direct Waves (from TX to RX)
[xt_d, xr_d] = meshgrid(TxRx(:,1), TxRx(:,1));
[yt_d, yr_d] = meshgrid(TxRx(:,2), TxRx(:,2));
distTxRx = sqrt((xt_d-xr_d).^2 + (yt_d-yr_d).^2);
E_d =   (1i/4)*besselh(0,1,k0*distTxRx);  % Ns x Ni
% E_d = exp(1i*k0*distTxRx)./(4*pi*distTxRx);  % Can try for 3D
clear xt_d xp_d yt_d yp_d G_Rx_Tx G_Tx_Rx Theta_Tx_Rx Theta_Rx_Tx

%% Definition of the Incident Waves (from TX to domain of interest)
[xt, xp] = meshgrid(TxRx(:,1), x(:));
[yt, yp] = meshgrid(TxRx(:,2), y(:));
distTxRn = sqrt((xt-xp).^2 + (yt-yp).^2);
E_inc =   (1i/4)*besselh(0,1,k0*distTxRn);  % M^2 x Ni for cylindrical wave
% E_inc = exp(1i*k0*distTxRn)./(4*pi*distTxRn);  % Can try for 3D
clear xt yt xp yp Theta_Tx_Rn Ref_theta_Tx_Rn G_Tx_Rn 

%% Estimate xRPI-LM Kernel
Fryt = zeros(length(Nodes(1,:)), M^2);
Fryt_k = zeros(length(Nodes(1,:)), M^2);
[xr, xpr] = meshgrid(TxRx(:,1), x(:));
[yr, ypr] = meshgrid(TxRx(:,2), y(:));
distRxRn = (sqrt((xr-xpr).^2 + (yr-ypr).^2))';
Zryt=((1i*pi*cellrad/(2*k0))*...
    besselj(1,k0*cellrad)*...
    besselh(0,1,k0*distRxRn)); % Integral of Greens
for i = 1:length(Nodes(1,:))
    Fryt(i,:) = ((k0^2)*((Zryt(Nodes(2,i),:).*...
        (E_inc(:,Nodes(1,i))).')./...
        (E_d(Nodes(2,i), Nodes(1,i)))));
end
% xRPI-LM Kernel Matrix:
FrytB =  [real(Fryt) -imag(Fryt)]; 





%% Data load and Reconstruction
%{
Description: Before this section, all the sections are for defining global
constants and fields. Next three sections are for loading the data, solving
the inverse problem and displaying the reconstruction. Therefore, the next
three sections can be put in the loop to load data-frames in real-time and
track movement of objects inside the imaging region.
%}
tic % Start the timer to evaluate time taken to process one dataset

%% Load Simulation or Experiment RSSI data (with target object and without target object)
if strcmp(Data_type,'Sim')      % Load Simulation data generated by Forward_data_simulation.m
    load ('Pinc.mat','Pinc')
    load ('Ptot.mat','Ptot')
end
if strcmp(Data_type,'Expr')     % Load experiment data from folder: "Experiment_data"
    load ('Experiment_data/Pinc.mat','Pinc')
    load ('Experiment_data/Ptot.mat','Ptot')
end
Ptot = Ptot(:);
Pinc=Pinc(:);
%% Solve the inverse Problem
Pryt=(Ptot-Pinc)./(20*log10(exp(1)));
    O = TVAL3(FrytB,Pryt,M,2*M,opts,lambX,lambY);
    Oimag = O(1:M,M+1:2*M);
epr = 4*pi*(Oimag*0.5)./lambda;
%% Reconstruction
figure(1)
imagesc(tx,ty,epr); hold on;
colormap jet
view([0 -90])
colorbar
ax = gca;
ax.FontSize = 18;
hold off;
time_elapsed = toc