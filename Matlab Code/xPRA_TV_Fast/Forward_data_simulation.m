%% Description
%{
Description: Simulates Forward 2D MoM case without conjugate gradient approach. 
Also includes directivity and transmit power. Program is benchmarked with SOM 
code of Prof Xudong where Inc, total and scat fields are all exactly matched 
as both codes uses Hankel func of first kind. Code is also benchmarked with
Eigenexpansion code where only magnitude and real part of Inc, total and
scat fields exactly matches. The imaginary parts are conjugate of each
other.
- Tx and Rx at same location and singular field is made zero
%}
%%
clear;
close all;
tic
imp = 120*pi;
geom = 'square';
NodeMode=1; NaNremove=1;
%% Method of Moment
freq = 2.4e9; lambda = 3e8/freq;  
k0 = 2*pi/lambda; 
save ('freq.mat','freq')
%% Parameters (keep unchanged for both forward problem and inverse problem)
size_DOI = 3;                 %round(2*lambda/0.75,1); 0.4  %;%0.24; %;          % size of DOI
epsono_r_c = 4+0.4*1i;               % the constant relative permittivity of the object
Ns = 20;                       % number of RXs
Ni = Ns;                        % number of TXs
M = 400;               % the square containing the object has a dimension of MxM
NumRes = M*lambda/(sqrt(real(epsono_r_c))*size_DOI);
xm=linspace(-size_DOI/2,size_DOI/2,Ns/4+1); % Node x positions
ym=linspace(-size_DOI/2,size_DOI/2,Ns/4+1); % Node y positions
%% Node Geometry
if strcmp(geom,'square')
XYRx = [[xm(1,:)', repmat(ym(1),[1,length(xm)])'];...
    [repmat(xm(1,end), [1,length(ym)-1])', ym(1,2:end)']; ...
    [flip(xm(1,1:end-1))', repmat(ym(1,end),[1,length(xm)-1])'];...
    [repmat(xm(1,1), [1,length(ym)-2])', flip(ym(1,2:end-1))']];
XYTx = [[xm(1,:)', repmat(ym(1),[1,length(xm)])'];...
    [repmat(xm(1,end), [1,length(ym)-1])', ym(1,2:end)']; ...
    [flip(xm(1,1:end-1))', repmat(ym(1,end),[1,length(xm)-1])'];...
    [repmat(xm(1,1), [1,length(ym)-2])', flip(ym(1,2:end-1))']];
X=XYRx(:,1)';Y=XYRx(:,2)';% Location of Rxs
X_tx=XYTx(:,1)';Y_tx=XYTx(:,2)';% Location of Txs
Rx_pos = [X(:), Y(:)];
Tx_pos = [X_tx(:), Y_tx(:)];
end
%% Positions of the cells
d = (size_DOI)/M;   % change     %the nearest distance of two cell centers
% tx = d/2:d:size_DOI-d/2;
% ty = size_DOI-d/2:-d:d/2;
tx = [d/2:d:size_DOI-d/2]-size_DOI/2;
ty = [size_DOI-d/2:-d:d/2]-size_DOI/2;
[x,y] = meshgrid(tx,ty);  % change                % M x M
cellrad = (sqrt(d^2/pi)*2)/2; % change                     % diameter of cells
%% Relative permittivity of each cell (Target Object)
epsono_r = ones(M,M);    %ones(M,M);  % change
obj_rad = 0.2;
epsono_r((x-0.12).^2+(y+0.12).^2<=(obj_rad)^2) = epsono_r_c;
%%
objInd = find((epsono_r-1)); NoobjInd = find((epsono_r-1)==0);
if (length(objInd)+length(NoobjInd))~= length(epsono_r(:))
   error('Cannot calculate with given values')
end
Z=zeros(length(objInd),length(objInd));  %[r, c]=ind2sub([m1,n1],objInd);
xObj = x(objInd); yObj = y(objInd);
for inc = 1:length(objInd) % first incident point in object
    xinc = x(objInd(inc)); yinc = y(objInd(inc)); %[r1,c1] = ind2sub([m1,n1],objInd(inc));    
    R=sqrt((xinc-xObj).^2 + (yinc-yObj).^2); %rhodd means distance from a unit dipole to another
    Z1 = -imp*pi*cellrad/2*besselj(1,k0*cellrad)*besselh(0,1,k0*R);
    Z1(inc) = -imp*pi*cellrad/2*besselh(1,1,k0*cellrad)-1i*imp*epsono_r(objInd(inc))/(k0*(epsono_r(objInd(inc))-1));
    Z(inc,:) = Z1.'; %Z1'
end
clear xinc yinc Z1 R yObj xObj
%% Definition of the Direct Waves (from TX to RX)
[xt_d, xr_d] = meshgrid(X_tx, X);
[yt_d, yr_d] = meshgrid(Y_tx, Y);
dist_tx_rx = sqrt((xt_d-xr_d).^2 + (yt_d-yr_d).^2);
E_d =  (1i/4)*besselh(0,1,k0*dist_tx_rx);  % Ns x Ni
% E_d =   exp(1i*k0*dist_tx_rx)./(4*pi*dist_tx_rx);  % Ns x Ni % Note that nth column is nth Tx and for each Tx, rows are Rx
clear xt_d xp_d yt_d yp_d dist_tx_rx G_Rx_Tx G_Tx_Rx Theta_Tx_Rx Theta_Rx_Tx
%% Definition of the Incident Waves (from TX to domain of interest)
[xt, xp] = meshgrid(X_tx, x(:));
[yt, yp] = meshgrid(Y_tx, y(:));
dist_tx_pix = sqrt((xt-xp).^2 + (yt-yp).^2);
E_inc =   (1i/4)*besselh(0,1,k0*dist_tx_pix);  % M^2 x Ni for cylindrical wave
% E_inc = exp(1i*k0*dist_tx_pix)./(4*pi*dist_tx_pix);  % M^2 x Ni for cylindrical wave
% % % % % % % % % % % % % % % 
clear xt yt xp yp dist_tx_pix Theta_Tx_Rn Ref_theta_Tx_Rn G_Tx_Rn
%% Using Matrix Inversion
E_inc_obj = -E_inc(objInd,:);
% Etot_rn=E_inc; 
% Einc = N x N_t
% Z = N x N
% J = N x Nt
% Z=gpuArray(Z);
% E_inc_obj=gpuArray(E_inc_obj);
J1=Z\(E_inc_obj);
% J1=gather(J1);
J = zeros(M^2,Ni);
for ii=1:length(objInd)
    J(objInd(ii),:) = J1(ii,:);
%     Etot_rn(objInd(ii),:) = 1i*imp*(J1(ii,:)./((epsono_r(objInd(ii))-1)*k0));    
end
clear Z J1
%% Generate Scattered E field
[xr, xpr] = meshgrid(X, x(:));
[yr, ypr] = meshgrid(Y, y(:));
dist_rx_pix = sqrt((xr-xpr).^2 + (yr-ypr).^2);
ZZ = -imp*pi*cellrad/2*besselj(1,k0*cellrad)*besselh(0,1,k0*dist_rx_pix'); % Ns x M^2
E_s = ZZ*J; % Ns x Ni
clear dist_rx_pix xr J yr xpr ypr G_Rx_Rn Theta_Rx_Rn ZZ
%%
if NodeMode==0 % nodes are not transreceivers
TxRxpairs=[];
for iii=1:length(Tx_pos(:,1))
    for jjj=1:length(Rx_pos(:,1))
        TxRxpairs= [TxRxpairs; iii,jjj];
    end
end
end
if NodeMode==1
TxRxpairs=[];
for iii=1:length(Tx_pos(:,1))
    for jjj=1:length(Rx_pos(:,1))
        if iii~=jjj
        TxRxpairs= [TxRxpairs; iii,jjj];
        end
    end
end
end
TxRxpairs=TxRxpairs';
%% Remove NaN
E_ds = E_d + E_s; % Ns x Ni; total field at RX = direct wave + scattered wave
E_ds1 = E_ds;
E_d1 = E_d;
if NodeMode==1
    if NaNremove ==1    
    E_d(logical(eye(size(E_d)))) = []; % Or A = A(~eye(size(A)))
    E_d = reshape(E_d, Ns-1,Ni); 
    E_ds(logical(eye(size(E_ds)))) = []; % Or A = A(~eye(size(A)))
    E_ds = reshape(E_ds, Ns-1,Ni); 

    E_s(logical(eye(size(E_s)))) = []; % Or A = A(~eye(size(A)))
    E_s = reshape(E_s, Ns-1,Ni);     
    end
    if NaNremove==0
    E_d(logical(eye(size(E_d)))) = 0; % Or A = A(~eye(size(A)))
    E_d = reshape(E_d, Ns,Ni); 
    E_ds(logical(eye(size(E_ds)))) = 0; % Or A = A(~eye(size(A)))
    E_ds = reshape(E_ds, Ns,Ni); 
    end    
end
TxRx1=Rx_pos;
%% Save data
F_ds = abs(E_ds);           % Ns x Ni; amplitude of total field at RX
timee = toc;

Pinc = ((abs(E_d)).^2)*(lambda*lambda)/(4*pi*imp);
Pinc = 10*log10(Pinc/1e-3);
Ptot = (F_ds.^2)*(lambda*lambda)/(4*pi*imp);
Ptot = 10*log10(Ptot/1e-3);

TxRx1=Tx_pos;
save ('TxRxpairs.mat','TxRxpairs')
save ('Pinc.mat','Pinc')
save ('Ptot.mat','Ptot')
save ('TxRx1.mat','TxRx1')
save ('xm.mat','xm')
save ('ym.mat','ym')
save ('Ni.mat','Ni')
save ('E_d.mat','E_d')
save ('E_ds1.mat','E_ds1')
save ('E_s.mat','E_s')
save ('epsono_r_c.mat','epsono_r_c')
%% New DOI
epsono_rGT = epsono_r; tx1 = tx; ty1=ty;
save ('size_DOI.mat','size_DOI')
save ('epsono_rGT.mat','epsono_rGT')
save ('tx1.mat','tx1')
save ('ty1.mat','ty1')
%% plot
figure(1); 
imagesc(tx,ty,imag(epsono_r)); hold on;
set(gca,'YDir','normal');
% set(gcf,'color','w');
colormap(jet);
xlim([-max(xm) max(xm)])
ylim([-max(ym) max(ym)])
set(gca,'FontSize',14)
% axis equal
xticks([-max(xm):0.6:max(xm)])
yticks([-max(xm):0.6:max(xm)])
% colormap(gray);
title('DOI with scattering object and wireless nodes')
xlabel('meters') 
ylabel('meters') 
colorbar;
scatter(X', Y', 100,'o','filled'); 
% scatter(X_tx', Y_tx', 60,'o','filled'); 
grid on;
hold off;

figure(2)
plot(1:Ns-1, F_ds(:,Ni), 'b', 1:Ns-1, abs(E_d(:,Ni)), 'r',...
    1:Ns-1, abs(E_s(:,Ni)), 'k')
title('Simulated field')
legend('Total','Inc','Scattered')