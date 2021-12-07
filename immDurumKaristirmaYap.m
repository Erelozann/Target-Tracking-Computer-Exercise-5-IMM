
function [X0,P0] = immDurumKaristirmaYap(StateEstimates, StateEstimatesCov, Mij)
% Aciklama:
%--------------------------------------------------------------------------
%  IMM baslangicinda farkli modellere ait kestirim ve kestirim
%  kovaryanslari birbiriyle karistirilir.
%
% Input:
%-------------
% StateEstimates        : Durum kestirim vektoru
% StateEstimatesCov     : Durum kestirim degisinti matrisi
% Mij                   : Karistirma olasiligi
%
% Output:
%-------------
% X0                    : Oncul kestirim vektoru
% P0                    : Oncul kestirim kovaryans matrisi
%--------------------------------------------------------------------------

    % Ilgili degiskenler tanimlanir:
    %--------------------------------
    nModels = size(StateEstimates,2);
    nState = size(StateEstimates,1);

    % Karistirilmis oncul kestirim vektoru olusturulur:
    %---------------------------------------------------
    X0 = zeros(size(StateEstimates));
    for i = 1:nModels
        X0(:,i) = sum((StateEstimates.*repmat(Mij(:,i)',nState,1)),2); 
    end
    
    % Karistirilmis oncul kestirim kovaryans matrisi olusturulur:
    %---------------------------------------------------------------
    P0 = zeros(size(StateEstimatesCov));
    for i=1:nModels
        for j=1:nModels
            P0(:,:,i) = P0(:,:,i) + Mij(j,i)*(StateEstimatesCov(:,:,j) + (StateEstimates(:,j) - X0(:,i))*(StateEstimates(:,j) - X0(:,i))');
        end
    end

end