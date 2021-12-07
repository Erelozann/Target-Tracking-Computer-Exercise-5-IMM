
function [Mu] = immModelOlasiliginiGuncelle(L, c)
% Aciklama:
%--------------------------------------------------------------------------
%  Yapilmis ongorumler ile olcum arasinda hesaplanan likelihoodlar ile
%  normalize edilmis model olasiliklari kullanilarak IMM dongusu sonunda
%  model olasiliklari guncellenir.
%
% Input:
%-------------
% L                     : Yapýlmis ongorumun Likelihood'u
% c                     : Normalize mod olasilik katsayisi
%
% Output:
%-------------
% Mu                    : Guncellenmis mod olasiliklari
%--------------------------------------------------------------------------

        Cm = L*c;                % Normalization constant
        Mu = L'.*c*(1/Cm);       % Mode probability update


end