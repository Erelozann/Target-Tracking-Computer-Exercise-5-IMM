
function [combinedVektor, combinedMatris] = immDurumBirlestirmeYap(vektor, matris, Mu)
% Aciklama:
%--------------------------------------------------------------------------
%  Verilen model olasiklarina gore ilgili modellere ait durum/ongorum
%  vektorleri ve durum/ongorum kovaryans matrisleri birlestirilir.
%
% Input:
%-------------
% vektor                : Modellere ait durum/ongorum vektorleri
% matris                : Modellere ait durum/ongorum kovaryans matrisleri
% Mu                    : Mod olasiliklari
%
% Output:
%-------------
% combinedVektor        : Modlar arasi gecis olasilik matrisi
% combinedMatris        : Normalize mod olasilik katsayisi
%--------------------------------------------------------------------------
    
    % Degiskenler tanimlanir:
    %------------------------------------------------------
    nState = size(vektor,1);
    combinedMatris = zeros(nState,nState);

    % Durum/ongorum vektorleri birlestirilir:
    %------------------------------------------------
    combinedVektor = vektor*Mu;
    
    % Durum/ongorum kovaryans matrisleri birlestirilir:
    %------------------------------------------------------
    nModels = size(vektor,2);
    for modelNo = 1:nModels
        
        combinedMatris = combinedMatris + Mu(modelNo)*(matris(:,:,modelNo) + (vektor(:,modelNo) - combinedVektor)*(vektor(:,modelNo) - combinedVektor)'); 
        
    end

end