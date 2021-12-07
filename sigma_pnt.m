

function [x, W]= sigma_pnt(x_mean, x_cov, a)
% Aciklama:
%   �zel Noktalar'�n (Sigma Points) Hesaplanmas�
%--------------------------------------------------------------------------
% Input:
%-------------
% x_mean    :   Modellenecek olan Gauss da��l�m�n�n Ortalama matrisi.
% x_cov     :   Modellenecek olan Gauss da��l�m�n�n kovaryans matrisi.
% a         :   Serbest Parametre
%--------------------------------------------------------------------------
% Output:
%-------------
% x         :   Hesaplanan �zel Noktalar
% W         :   Hesaplanan �zel Noktalar�n A��rl�klar�
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
 
