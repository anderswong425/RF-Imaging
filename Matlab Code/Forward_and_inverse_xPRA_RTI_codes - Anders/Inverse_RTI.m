%% Parameters and Data
%{
Description: RTI code
%}
clear; close all; clc; 
tic
% load parameters from the simulation code
load ('Ni.mat','Ni')               % Number of sensor nodes = 20
load ('TxRxpairs.mat','TxRxpairs') % All possible pairs of Tx and Rx nodes (total 20(20-1)=380 pairs)
Nodes=TxRxpairs; 
clear TxRxpairs
load ('TxRx1.mat','TxRx1')         % Location of all 40 sensors
load ('size_DOI.mat','size_DOI')   % Size of DOI = 3x3 m^2

% Simulation RSSI data (with target object and without target object)
load ('Ptot.mat','Ptot')
load ('Pinc.mat','Pinc')  % 

TxRx = TxRx1; 
RoomLength = size_DOI; 
RoomWidth = size_DOI;
%% Convert Matrix to measurement vector
Ptot = Ptot(:);
Pinc=Pinc(:);
%% Important Global Paramters
gridXres = 0.05;         % grid/pixel/voxel resolution in x direction (in meters)
gridYres = gridXres;    % grid or pixel resolution in y direction (in meters)
Method = 'RTI';         % Select method
SemiMin =0.1;           % size of the ellipse used in RTI (in meters)
alph =  0.5;             % Regularization parameter for Ridge
p=RoomWidth/gridYres;   % number of grids in DOI in x direction
q=RoomLength/gridXres;  % number of grids in DOI in y direction
%% Create 2D Mesh of all the grids in DOI
tx = [gridXres/2:gridXres:RoomLength-gridXres/2]-RoomLength/2; %0:gridXres:2;%1:gridYres:2;%
ty = [RoomWidth-gridYres/2:-gridYres:gridYres/2]-RoomWidth/2;
[x,y] = meshgrid(tx,ty);
M =p;                   % total number of grids inside DOI = M^2
%% Distance from TXs to RXs
[xt_d, xr_d] = meshgrid(TxRx(:,1), TxRx(:,1));
[yt_d, yr_d] = meshgrid(TxRx(:,2), TxRx(:,2));
distTxRx = sqrt((xt_d-xr_d).^2 + (yt_d-yr_d).^2);

%% Distance from TXs to grids inside DOI
[xt, xp] = meshgrid(TxRx(:,1), x(:));
[yt, yp] = meshgrid(TxRx(:,2), y(:));
distTxRn = sqrt((xt-xp).^2 + (yt-yp).^2);
%% Implement RTI
F_RTI = zeros(length(Nodes(1,:)), M^2);
[xr, xpr] = meshgrid(TxRx(:,1), x(:));
[yr, ypr] = meshgrid(TxRx(:,2), y(:));
distRxRn = (sqrt((xr-xpr).^2 + (yr-ypr).^2))';   % distance of each grid from Receivers
if strcmp(Method,'RTI')
    for i = 1:length(Nodes(1,:))
        Thresh = 2*sqrt((distTxRx(Nodes(2,i), Nodes(1,i))^2)/4 + SemiMin^2);
        foc_sum = distRxRn(Nodes(2,i),:)+distTxRn(:,Nodes(1,i))';
        foc_sum(foc_sum > Thresh) = 0;   % Assign zero weight to all grids outside the ellipse
        foc_sum(foc_sum ~= 0) = 1;
        F_RTI(i,:) = foc_sum;%./sqrt(distTxRx(Nodes(2,i), Nodes(1,i))^2);
    end
    Pryt=-(Ptot-Pinc);
    O = ((F_RTI'*F_RTI + alph*eye(M^2,M^2))\F_RTI'*Pryt);
%     O = (O(:)./max(O(:))); % Normalize the solution
    O(O<=0) = 0;           % Remove values smaller than zero (can also do it by solving constrained optimization problem)
    O = reshape(O, p,q);
end
clear xr yr xpr ypr G_Rx_Rn Theta_Rx_Rn
clear  TxRxpairs distRxRn distTxRn
%% PLOT 

figure(3)
epr_real = (O);
imagesc(tx,ty,epr_real); hold on;
colormap jet
view([0 -90])
ax = gca; ax.FontSize = 18; 
colorbar
hold off;fig = figure(3); movegui(fig,'center');
timeElapsed = toc