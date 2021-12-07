function [FusedCovarianceTrace,FusedModeProb,FusedCovariance,FusedEstimate] = SPCF(GaussianModelProb, GaussianStateEstimate, GaussianStateEstimateCov, MixtureModelProbs, MixtureStateEstimates, MixtureStateEstimatesCov, w)

N = size(MixtureModelProbs,1);

FusedCovariance = zeros(size(GaussianStateEstimateCov));

a = det(2*pi*w^-1*GaussianStateEstimateCov)^0.5/det(2*pi*GaussianStateEstimateCov)^(w/2);

[Beta] = GaussianMixturePowerTaker(1-w,MixtureModelProbs,MixtureStateEstimates,MixtureStateEstimatesCov);

d = zeros(N,1);
c = 0;

CondCov = zeros(size(MixtureStateEstimatesCov));
CondEst = zeros(size(MixtureStateEstimates));

for i=1:N

InfoGaussian = w*inv(GaussianStateEstimateCov);
InfoMixture = (1-w)*inv(MixtureStateEstimatesCov(:,:,i));
CondCov(:,:,i) = inv(InfoGaussian+InfoMixture);
CondEst(:,i) = CondCov(:,:,i)*(InfoGaussian*GaussianStateEstimate+InfoMixture*MixtureStateEstimates(:,i)); 

Cov = w^-1*GaussianStateEstimateCov + (1-w)^-1*MixtureStateEstimatesCov(:,:,i);
d(i) = Beta(i)*GaussianDensity(MixtureStateEstimates(:,i),GaussianStateEstimate,Cov);
c = c + d(i);

end

ita = d/c;
FusedModeProb = GaussianModelProb^w*a*c;

FusedEstimate = CondEst*ita;

for i=1:N


PartialCovariance = ita(i)*(CondCov(:,:,i) + (CondEst(:,i)-FusedEstimate)*(CondEst(:,i)-FusedEstimate)');

FusedCovariance = FusedCovariance + PartialCovariance;

end

FusedCovarianceTrace = trace(FusedCovariance);