function [D,Dt] = defDDt(lambX,lambY)

D = @(U) ForwardD(U,lambX,lambY);
Dt = @(X,Y) Dive(X,Y,lambX,lambY);

function [Dux,Duy] = ForwardD(U,lambX,lambY)
% [ux,uy] = D u

Dux = lambX*[diff(U,1,2), U(:,1) - U(:,end)];
Duy = lambY*[diff(U,1,1); U(1,:) - U(end,:)];

function DtXY = Dive(X,Y,lambX,lambY)
% DtXY = D_1' X + D_2' Y

DtXY = lambX*[X(:,end) - X(:, 1), -diff(X,1,2)];
DtXY = DtXY + lambY*[Y(end,:) - Y(1, :); -diff(Y,1,1)];
DtXY = DtXY(:);
















%% Original
% function [Dux,Duy] = ForwardD(U,lambX,lambY)
% % [ux,uy] = D u
% 
% Dux = lambX*[diff(U,1,2), U(:,1) - U(:,end)];
% Duy = lambY*[diff(U,1,1); U(1,:) - U(end,:)];
% 
% function DtXY = Dive(X,Y,lambX,lambY)
% % DtXY = D_1' X + D_2' Y
% 
% DtXY = lambX*[X(:,end) - X(:, 1), -diff(X,1,2)];
% DtXY = DtXY + lambY*[Y(end,:) - Y(1, :); -diff(Y,1,1)];
% DtXY = DtXY(:);