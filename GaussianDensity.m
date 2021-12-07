function Density = GaussianDensity(Point,Mean,Covariance)

dif = Point-Mean;
mahalanobisUzaklik = dif'/Covariance*dif;

Density = exp(-mahalanobisUzaklik/2)/sqrt(det(2*pi*Covariance));