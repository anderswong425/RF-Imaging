%% Description
%{
Description: 
%}
clear; close all; clc; 
tic
load ('TxRxpairs.mat','TxRxpairs')
Nodes=TxRxpairs; clear TxRxpairs
load ('TxRx1.mat','TxRx1')
load ('size_DOI.mat','size_DOI')
load ('xm.mat','xm')
load ('ym.mat','ym')
load ('Ni.mat','Ni')
clear opts
// Pt = 19.5; Ct=1; Cr=1;
TxRx = TxRx1; RoomLength = size_DOI; RoomWidth = size_DOI;
imp = 120*pi; c = 3e8; freq = 2.4e9; lambda = c/freq; k0 = 2*pi/lambda;   
%% RSSI Data
load('Pinc.mat','Pinc')
load('Ptot.mat','Ptot')
load('E_ds1.mat','E_ds1');
E_ds1 = sqrt((10.^(Ptot/10))*(1e-3)*4*pi*imp/lambda^2);  
E_ds=E_ds1;
load('E_d.mat','E_d') 
load('E_s.mat','E_s')
%% Important Global Paramters
gridXres = 0.05;  gridYres = gridXres;  %  Resolution of grid
Method = 'xPRA';  % RTI xPRAI xPRA
Methodreg = 'ridge'; % 'tv'    'ridge' 'H1'
Eterm = 1;
alph = 0.5; 
sf=2; normz=0;
%% Global Paramters
lambX=1; lambY=1;
p=RoomWidth/gridYres; q=RoomLength/gridXres;
%%
tx = [gridXres/2:gridXres:RoomLength-gridXres/2]-RoomLength/2;%0:gridXres:2;%1:gridYres:2;%
ty = [RoomWidth-gridYres/2:-gridYres:gridYres/2]-RoomWidth/2;
[x,y] = meshgrid(tx,ty);
M =length(tx);
%% Constants                                                                    % Correction factor                                                     % Free Space permittivity
Gridarea = gridXres*gridYres;  Gridvol = Gridarea*gridYres;                                              % Area of one grid
cellrad = (sqrt(gridXres^2/pi)*2)/2; % radius of cells
Gridarea=(4*pi*cellrad/(2*k0))*besselj(1,k0*cellrad);
%% Total Received power (scattering+Incident)
Ptot = Ptot(:);
Pinc=Pinc(:);
%% Definition of the Direct Waves (from TX to RX)
[xt_d, xr_d] = meshgrid(TxRx(:,1), TxRx(:,1));
[yt_d, yr_d] = meshgrid(TxRx(:,2), TxRx(:,2));
distTxRx = sqrt((xt_d-xr_d).^2 + (yt_d-yr_d).^2);
E_d =   (1i/4)*besselh(0,1,k0*distTxRx);  % Ns x Ni
% E_d = exp(1i*k0*distTxRx)./(4*pi*distTxRx);  % M^2 x Ni for cylindrical wave
clear xt_d xp_d yt_d yp_d G_Rx_Tx G_Tx_Rx Theta_Tx_Rx Theta_Rx_Tx
%% Definition of the Incident Waves (from TX to domain of interest)
[xt, xp] = meshgrid(TxRx(:,1), x(:));
[yt, yp] = meshgrid(TxRx(:,2), y(:));
distTxRn = sqrt((xt-xp).^2 + (yt-yp).^2);
E_inc =   (1i/4)*besselh(0,1,k0*distTxRn);  % M^2 x Ni for cylindrical wave
% E_inc = exp(1i*k0*distTxRn)./(4*pi*distTxRn);  % M^2 x Ni for cylindrical wave
clear xt yt xp yp Theta_Tx_Rn Ref_theta_Tx_Rn G_Tx_Rn 

%% Implement xPRA
Fryt = zeros(length(Nodes(1,:)), M^2);
// Fryt_k = zeros(length(Nodes(1,:)), M^2);
[xr, xpr] = meshgrid(TxRx(:,1), x(:));
[yr, ypr] = meshgrid(TxRx(:,2), y(:));
distRxRn = (sqrt((xr-xpr).^2 + (yr-ypr).^2))';
if strcmp(Method,'xPRA')
    Zryt=((1i*pi*cellrad/(2*k0))*...
        besselj(1,k0*cellrad)*...
        besselh(0,1,k0*distRxRn)); % Integral of Greens
    for i = 1:length(Nodes(1,:))
        Fryt(i,:) = ((k0^2)*((Zryt(Nodes(2,i),:).*...
            (E_inc(:,Nodes(1,i))).')./...
            (E_d(Nodes(2,i), Nodes(1,i)))));
    end
    E_d1=E_d;
    E_d1(logical(eye(size(E_d1)))) = []; % Or A = A(~eye(size(A)))
    E_d1 = reshape(E_d1, Ni-1,Ni);
    Pryt=(Ptot-Pinc)./(20*log10(exp(1)));
    % Optimize
    FrytB =  [real(Fryt) -imag(Fryt)];
    if strcmp(Methodreg,'ridge')
        U_l = length(FrytB(1,:));
        lambda_max = norm( FrytB'*Pryt, 2);
        O = (FrytB.'*FrytB + lambda_max*alph*eye(U_l))\FrytB.'*Pryt;
        Oreal = reshape(O(1:M^2,1), p,q);
        Oimag = reshape(O(M^2+1:2*M^2,1), p,q);
    end
    epr = 4*pi*(Oimag*0.5)./lambda; 
end

clear xr yr xpr ypr G_Rx_Rn Theta_Rx_Rn
clear  TxRxpairs
%% PLOT
epr(epr<0)=0;
%% Reconstruction
figure(8)
    imagesc(tx,ty,Eterm*epr); hold on;
    colormap jet
    view([0 -90])
    colorbar
        ax = gca;
    ax.FontSize = 18;
%     axis('square')
    title([(Method) ]); 
 hold off;

%  figure(9)
%     imagesc(tx,ty,chi_i_GT); hold on;
%     colormap jet
%     view([0 -90])
%     colorbar
%     xlim([-size_DOI/sf size_DOI/sf])
%     ylim([-size_DOI/sf size_DOI/sf])
%         ax = gca;
%     ax.FontSize = 18;
% %     axis('square')
%     title('Ground Truth', 'FontSize', 12);
%  hold off;