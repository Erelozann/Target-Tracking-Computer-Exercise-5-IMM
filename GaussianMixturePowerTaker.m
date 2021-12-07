function [NewModelProbabilities] = GaussianMixturePowerTaker(w,ModelProbabilities,StateEstimates,StateEstimatesCov)

NewStateEstimatesCov = (w^-1)*StateEstimatesCov;
N = size(ModelProbabilities,1);
n = size(StateEstimates,1);
SigPntNumber = 2*n+1;

M = zeros(N*SigPntNumber,N);
b = zeros(N*SigPntNumber,1);
W = zeros(N*SigPntNumber);

for i = 1:N
    alpha = ModelProbabilities(i);
    [s_j,pi_j] = sigma_pnt(StateEstimates(:,i),StateEstimatesCov(:,:,i),1e-3);
    for j=1:SigPntNumber
        W(SigPntNumber*(i-1)+j,SigPntNumber*(i-1)+j) = alpha*pi_j(j);
        p = 0;
        for m = 1:N
            p = p + ModelProbabilities(m)*GaussianDensity(s_j(:,j),StateEstimates(:,m),StateEstimatesCov(:,:,m));
            M(SigPntNumber*(i-1)+j,m) = GaussianDensity(s_j(:,j),StateEstimates(:,m),NewStateEstimatesCov(:,:,m));
        end
        b(SigPntNumber*(i-1)+j,1) = p^w;
    end
end

R = chol(W);
d = R*b;
C = R*M;
x = lsqnonneg(C,d);
NewModelProbabilities = x;