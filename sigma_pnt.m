

function [x, W]= sigma_pnt(x_mean, x_cov, a)
% Aciklama:
%   Özel Noktalar'ýn (Sigma Points) Hesaplanmasý
%--------------------------------------------------------------------------
% Input:
%-------------
% x_mean    :   Modellenecek olan Gauss daðýlýmýnýn Ortalama matrisi.
% x_cov     :   Modellenecek olan Gauss daðýlýmýnýn kovaryans matrisi.
% a         :   Serbest Parametre
%--------------------------------------------------------------------------
% Output:
%-------------
% x         :   Hesaplanan Özel Noktalar
% W         :   Hesaplanan Özel Noktalarýn Aðýrlýklarý
%--------------------------------------------------------------------------

    % Degiskenler tanimlanir:
    %--------------------------
    N = length(x_mean);
    W = zeros(1,2*N+1);

    x(:,1) = x_mean;
    W(1) = a;

    A = (N/(1-a))*x_cov;

    delta_x = chol(real(A),'lower');

    for i=2:(N+1)
        W(i) = (1-a)/(2*N);
        W(i+N) = (1-a)/(2*N);

        x(:,i) = x_mean + delta_x(:,i-1);
        x(:,i+N) = x_mean - delta_x(:,i-1);
    end

end
 
