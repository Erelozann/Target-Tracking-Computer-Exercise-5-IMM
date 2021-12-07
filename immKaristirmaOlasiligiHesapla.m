
function [p, c, Mij] = immKaristirmaOlasiligiHesapla(trans_pro, Ts, Mu)
% Aciklama:
%--------------------------------------------------------------------------
%  IMM baslangicinda farkli modellere ait kestirim ve kestirim
%  kovaryanslari birbiriyle karistirilir.
%
% Input:
%-------------
% Ts                    : Son guncellemeden beri gecen sure (s)
% Mu                    : Mod olasiliklari
%
% Output:
%-------------
% p                     : Modlar arasi gecis olasilik matrisi
% c                     : Normalize mod olasilik katsayisi
% Mij                   : Karistirma olasiligi
%--------------------------------------------------------------------------

    p = trans_pro;

    [I] = size(p,1);
    p = p^abs(Ts);
    c = p'*Mu; 
    Mij = (p.*repmat(Mu,1,I))./repmat((c+eps)',I,1); 
    
end