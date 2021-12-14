function [FusedModeProb,FusedCovariance,FusedEstimate] = NaiveFusion(GaussianModelProb, GaussianStateEstimate, GaussianStateEstimateCov, MixtureModelProbs, MixtureStateEstimates, MixtureStateEstimatesCov)

N = size(MixtureModelProbs,1);

FusedCovariance = zeros(size(GaussianStateEstimateCov));

d = zeros(N,1);
c = 0;

CondCov = zeros(size(MixtureStateEstimatesCov));
CondEst = zeros(size(MixtureStateEstimates));

for i=1:N

InfoGaussian = inv(GaussianStateEstimateCov);
InfoMixture = inv(MixtureStateEstimatesCov(:,:,i));
CondCov(:,:,i) = inv(InfoGaussian+InfoMixture);
CondEst(:,i) = CondCov(:,:,i)*(InfoGaussian*GaussianStateEstimate+InfoMixture*MixtureStateEstimates(:,i)); 

Cov = GaussianStateEstimateCov + MixtureStateEstimatesCov(:,:,i);
d(i) = MixtureModelProbs(i)*GaussianDensity(MixtureStateEstimates(:,i),GaussianStateEstimate,Cov);
c = c + d(i);

end

ita = d/c;
FusedModeProb = GaussianModelProb*c;

FusedEstimate = CondEst*ita;

for i=1:N

PartialCovariance = ita(i)*(CondCov(:,:,i) + (CondEst(:,i)-FusedEstimate)*(CondEst(:,i)-FusedEstimate)');

FusedCovariance = FusedCovariance + PartialCovariance;

end