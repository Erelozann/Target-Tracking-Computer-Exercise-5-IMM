%% 2 Track Fusion

%% Initialization
% True Target Model

T = 1; %Sample Interval

% State Transition Model

F = [eye(2)  T*eye(2); %State Transition Matrix
     zeros(2) eye(2)];

G = [T^2/2*eye(2); %Process Noise Gain Matrix
     T*eye(2)];

sigma_a = 2; %Acceleration Noise Standard Deviation
Q = (sigma_a)^2*eye(2); %Process Noise Covariance


Q_coeffs = [1 10];

TransProb(:,:,1) = [0.9 0.1;
                    0.1 0.9];

% Measurement Model 

H = [eye(2) zeros(2)]; %Measurement Matrix

sigma_v = 20; %Measurement Noise Standard Deviation
R = (sigma_v)^2*eye(2); %Measurement Noise Covariance
% Simulation Parameters

Step_Num = 100;
MC_Num = 100; %Monte Carlo Number
Model_Num = 2;

State = zeros(Step_Num,MC_Num);
State(1,:) = 1;

x0_bar = [5e3 5e3 25 25]';
P0_bar = diag((x0_bar/10).^2);


x_k = zeros(4,Step_Num,MC_Num);
y_k_1 = zeros(2,Step_Num,MC_Num);
y_k_2 = zeros(2,Step_Num,MC_Num);

chol_R = chol(R);
chol_P0_bar = chol(P0_bar);

for MC = 1:MC_Num
 
 x_k(:,1,MC) = x0_bar + chol_P0_bar*randn(4,1); % True Target Data Generation
 y_k_1(:,1) = H*x_k(:,1,MC)+ chol_R*randn(2,1);
 y_k_2(:,1) = H*x_k(:,1,MC)+ chol_R*randn(2,1);
    for i=2:Step_Num
        CurrentState = State(i-1,MC);
        x_k(:,i,MC) = F*x_k(:,i-1,MC) + G*(Q_coeffs(CurrentState)*sigma_a*randn(2,1)); % True Target Data Generation
        y_k_1(:,i,MC) = H*x_k(:,i,MC)+ chol_R*randn(2,1);
        y_k_2(:,i,MC) = H*x_k(:,i,MC)+ chol_R*randn(2,1);
        if rand > TransProb(CurrentState,CurrentState)
            if CurrentState == 1
                State(i,MC) = 2;
            else
                State(i,MC) = 1;
            end
        else
             State(i,MC) = CurrentState;
        end
    end
end

yk_centralized = [y_k_1;
                  y_k_2];
              
%% Centralized Solution 

eps_centralized = zeros(MC_Num,Step_Num); % Normalized Estimation Error Squares Matrix

% Filter Parameters

TransProb = [0.9 0.1;
             0.1 0.9];

IMM_Filter_Parameters(1).H= [H;H];                      
IMM_Filter_Parameters(1).Q = Q;
IMM_Filter_Parameters(1).R = blkdiag(R,R);
IMM_Filter_Parameters(1).T = T;

IMM_Filter_Parameters(2).H= [H;H];                     
IMM_Filter_Parameters(2).Q = 100*Q;
IMM_Filter_Parameters(2).R = blkdiag(R,R);
IMM_Filter_Parameters(2).T = T;


% Estimate Initialization
StateEstimates = zeros(4,Model_Num,Step_Num);
CombinedStateEstimate = zeros(4,Step_Num);
StateEstimates(:,1,1) = x0_bar;
StateEstimates(:,2,1) = x0_bar;
CombinedStateEstimate(:,1) = x0_bar;

StateEstimatesCov = zeros(4,4,Model_Num,Step_Num);
StateEstimatesCov(:,:,1,1) = P0_bar;
StateEstimatesCov(:,:,2,1) = P0_bar;
CombinedStateEstimateCov = zeros(4,4,Step_Num);
CombinedStateEstimateCov(:,:,1) = P0_bar;
Mu = zeros(Model_Num,Step_Num);
Mu(:,1) = [0.5;0.5];
TransProb = [0.9 0.1;
             0.1 0.9];

err_cent = zeros(4,Step_Num,MC_Num);

% Monte Carlo Simulation
StatePredictions = zeros(4,Model_Num);
StatePredictionsCov = zeros(4,4,Model_Num);
OutputPredictions = zeros(4,Model_Num);
OutputPredictionsCov = zeros(4,4,Model_Num);
KalmanGains = zeros(4,4,Model_Num);
L = zeros(1,Model_Num);

for MC = 1:MC_Num

    err_cent(:,1,MC) = x_k(:,1,MC)-CombinedStateEstimate(:,1);
    eps_centralized(MC,1) = err_cent(:,1,MC)'*inv(CombinedStateEstimateCov(:,:,1))*err_cent(:,1,MC);
    
    for i=1:Step_Num-1
        
        % Karistirma olasiliklari hesaplaniyor:
        %----------------------------------------
        [~, c, Mij] = immKaristirmaOlasiligiHesapla(TransProb, T, Mu(:,i));
        
        % Karistirma yapiliyor:
        %---------------------------
        [MixedStateEstimates,MixedStateEstimatesCov] = immDurumKaristirmaYap(StateEstimates(:,:,i), StateEstimatesCov(:,:,:,i), Mij);
        
        for k=1:Model_Num
        
        % Prediction:
        
        [StatePredictions(:,k), ...
            StatePredictionsCov(:,:,k), ...
            OutputPredictions(:,k), ...
            OutputPredictionsCov(:,:,k), ...
            KalmanGains(:,:,k)] = kf_pre(MixedStateEstimates(:,k),...
            MixedStateEstimatesCov(:,:,k),...
            IMM_Filter_Parameters(k));

        % Estimation:
        [...
            StateEstimates(:,k,i+1), ...
            StateEstimatesCov(:,:,k,i+1),...
            L(k)] = kf_est(...
            StatePredictions(:,k),...
            StatePredictionsCov(:,:,k),...
            OutputPredictions(:,k), ...
            OutputPredictionsCov(:,:,k), ...
            KalmanGains(:,:,k),...
            yk_centralized(:,i+1,MC));
        

        end
        
        Mu(:,i+1) = immModelOlasiliginiGuncelle(L, c);
        
        [CombinedStateEstimate(:,i+1), CombinedStateEstimateCov(:,:,i+1)] = immDurumBirlestirmeYap(StateEstimates(:,:,i+1), StateEstimatesCov(:,:,:,i+1), Mu(:,i+1));
        
        err_cent(:,i+1,MC) = x_k(:,i+1,MC)-CombinedStateEstimate(:,i+1);
        eps_centralized(MC,i+1) = err_cent(:,i+1,MC)'*inv(CombinedStateEstimateCov(:,:,i+1))*err_cent(:,i+1,MC);
        

    end
end


%% Decentralized Solution

% Filter Parameters
IMM_Filter_Parameters(1,1).H=  H;                      
IMM_Filter_Parameters(1,1).Q = Q_coeffs(1)^2*Q;
IMM_Filter_Parameters(1,1).R = R;
IMM_Filter_Parameters(1,1).T = T;

IMM_Filter_Parameters(1,2).H=  H;                     
IMM_Filter_Parameters(1,2).Q = Q_coeffs(2)^2*Q;
IMM_Filter_Parameters(1,2).R = R;
IMM_Filter_Parameters(1,2).T = T;

TransProb(:,:,1) = [0.9 0.1;
                    0.1 0.9];

IMM_Filter_Parameters(2,1).H=  H;                      
IMM_Filter_Parameters(2,1).Q = Q_coeffs(1)^2*Q;
IMM_Filter_Parameters(2,1).R = R;
IMM_Filter_Parameters(2,1).T = T;

IMM_Filter_Parameters(2,2).H=  H;                     
IMM_Filter_Parameters(2,2).Q = Q_coeffs(2)^2*Q;
IMM_Filter_Parameters(2,2).R = R;
IMM_Filter_Parameters(2,2).T = T;

TransProb(:,:,2) = [0.9 0.1;
                    0.1 0.9];
% Estimate Initialization

StateEstimates = zeros(4,Model_Num,Step_Num);
CombinedStateEstimate = zeros(4,Step_Num);
StateEstimates(:,1,1) = x0_bar;
StateEstimates(:,2,1) = x0_bar;
CombinedStateEstimate(:,1) = x0_bar;

StateEstimatesCov = zeros(4,4,Model_Num,Step_Num);
StateEstimatesCov(:,:,1,1) = P0_bar;
StateEstimatesCov(:,:,2,1) = P0_bar;
CombinedStateEstimateCov = zeros(4,4,Step_Num);
CombinedStateEstimateCov(:,:,1) = P0_bar;
Mu = zeros(Model_Num,Step_Num);
Mu(:,1) = [0.5;0.5];

Local_Trackers(1).IMM_Parameters = IMM_Filter_Parameters(1,:);
Local_Trackers(1).StateEstimates = StateEstimates;
Local_Trackers(1).CombinedStateEstimate = CombinedStateEstimate;
Local_Trackers(1).StateEstimatesCov = StateEstimatesCov;
Local_Trackers(1).CombinedStateEstimateCov = CombinedStateEstimateCov;
Local_Trackers(1).Mu = Mu;

Local_Trackers(2).IMM_Parameters = IMM_Filter_Parameters(2,:);
Local_Trackers(2).StateEstimates = StateEstimates;
Local_Trackers(2).CombinedStateEstimate = CombinedStateEstimate;
Local_Trackers(2).StateEstimatesCov = StateEstimatesCov;
Local_Trackers(2).CombinedStateEstimateCov = CombinedStateEstimateCov;
Local_Trackers(2).Mu = Mu;

StatePredictions = zeros(4,Model_Num);
StatePredictionsCov = zeros(4,4,Model_Num);
L = zeros(1,Model_Num);

eps_decentralized = zeros(MC_Num,Step_Num); % Normalized Estimation Error Squares Matrix

err_decent = zeros(4,Step_Num,MC_Num);

Fusion_Filter_Parameters.H = eye(4);
Fusion_Filter_Parameters.Q = Q;
Fusion_Filter_Parameters.T = 0;

for MC = 1:MC_Num
    for i=1:Step_Num-1
        
       if  i~= 1 && mod(i,2) == 1
            OutputPredictions = zeros(4,Model_Num);
            OutputPredictionsCov = zeros(4,4,Model_Num);
            KalmanGains = zeros(4,4,Model_Num);
            Fusion_Filter_Parameters.R = Local_Trackers(1).CombinedStateEstimateCov(:,:,i);
         
                
            for l=1:Model_Num
               
                % Prediction:
                
                [StatePredictions(:,l), ...
                    StatePredictionsCov(:,:,l), ...
                    OutputPredictions(:,l), ...
                    OutputPredictionsCov(:,:,l), ...
                    KalmanGains(:,:,l)] = kf_pre(Local_Trackers(2).StateEstimates(:,l,i),...
                    Local_Trackers(2).StateEstimatesCov(:,:,l,i),...
                    Fusion_Filter_Parameters);
                
                % Estimation:
                [...
                    Local_Trackers(2).StateEstimates(:,l,i), ...
                    Local_Trackers(2).StateEstimatesCov(:,:,l,i),...
                    L(l)] = kf_est(...
                    StatePredictions(:,l),...
                    StatePredictionsCov(:,:,l),...
                    OutputPredictions(:,l), ...
                    OutputPredictionsCov(:,:,l), ...
                    KalmanGains(:,:,l),...
                    Local_Trackers(1).CombinedStateEstimate(:,i));
        
            end
            
            Local_Trackers(2).Mu(:,i) = immModelOlasiliginiGuncelle(L, Local_Trackers(2).Mu(:,i));
            
            [Local_Trackers(2).CombinedStateEstimate(:,i), Local_Trackers(2).CombinedStateEstimateCov(:,:,i)] = immDurumBirlestirmeYap(Local_Trackers(2).StateEstimates(:,:,i), Local_Trackers(2).StateEstimatesCov(:,:,:,i),Local_Trackers(2).Mu(:,i));
            
        
        
        end
        
        err_decent(:,i,MC) = x_k(:,i,MC)-Local_Trackers(2).CombinedStateEstimate(:,i);
        eps_decentralized(MC,i) = err_decent(:,i,MC)'*inv(Local_Trackers(2).CombinedStateEstimateCov(:,:,i))*err_decent(:,i,MC);
        
        OutputPredictions = zeros(2,Model_Num);
        OutputPredictionsCov = zeros(2,2,Model_Num);
        KalmanGains = zeros(4,2,Model_Num);
        
        for k=1:2
            
         % Karistirma olasiliklari hesaplaniyor:
        %----------------------------------------
        [~, c, Mij] = immKaristirmaOlasiligiHesapla(TransProb(:,:,k), T, Local_Trackers(k).Mu(:,i));
        
        % Karistirma yapiliyor:
        %---------------------------
        [MixedStateEstimates,MixedStateEstimatesCov] = immDurumKaristirmaYap(Local_Trackers(k).StateEstimates(:,:,i), Local_Trackers(k).StateEstimatesCov(:,:,:,i), Mij);
            
            for l=1:Model_Num
                
                
                % Prediction:
                
                [StatePredictions(:,l), ...
                    StatePredictionsCov(:,:,l), ...
                    OutputPredictions(:,l), ...
                    OutputPredictionsCov(:,:,l), ...
                    KalmanGains(:,:,l)] = kf_pre(MixedStateEstimates(:,l),...
                    MixedStateEstimatesCov(:,:,l),...
                    Local_Trackers(k).IMM_Parameters(l));
                
                % Estimation:
                [...
                    Local_Trackers(k).StateEstimates(:,l,i+1), ...
                    Local_Trackers(k).StateEstimatesCov(:,:,l,i+1),...
                    L(l)] = kf_est(...
                    StatePredictions(:,l),...
                    StatePredictionsCov(:,:,l),...
                    OutputPredictions(:,l), ...
                    OutputPredictionsCov(:,:,l), ...
                    KalmanGains(:,:,l),...
                    yk_centralized(2*k-1:2*k,i+1,MC));
            end
            
            Local_Trackers(k).Mu(:,i+1) = immModelOlasiliginiGuncelle(L, c);
            
            [Local_Trackers(k).CombinedStateEstimate(:,i+1),Local_Trackers(k).CombinedStateEstimateCov(:,:,i+1)] = immDurumBirlestirmeYap(Local_Trackers(k).StateEstimates(:,:,i+1), Local_Trackers(k).StateEstimatesCov(:,:,:,i+1),Local_Trackers(k).Mu(:,i+1));
            
        end
    end
    
    err_decent(:,end,MC) = x_k(:,end,MC)-Local_Trackers(2).CombinedStateEstimate(:,end);
    eps_decentralized(MC,end) = err_decent(:,end,MC)'*inv(Local_Trackers(2).CombinedStateEstimateCov(:,:,end))*err_decent(:,end,MC);
       
end 

%% Covariance Intersection

% Filter Parameters
IMM_Filter_Parameters(1,1).H=  H;                      
IMM_Filter_Parameters(1,1).Q = Q_coeffs(1)^2*Q;
IMM_Filter_Parameters(1,1).R = R;
IMM_Filter_Parameters(1,1).T = T;

IMM_Filter_Parameters(1,2).H=  H;                     
IMM_Filter_Parameters(1,2).Q = Q_coeffs(2)^2*Q;
IMM_Filter_Parameters(1,2).R = R;
IMM_Filter_Parameters(1,2).T = T;

TransProb(:,:,1) = [0.9 0.1;
                    0.1 0.9];

IMM_Filter_Parameters(2,1).H=  H;                      
IMM_Filter_Parameters(2,1).Q = Q_coeffs(1)^2*Q;
IMM_Filter_Parameters(2,1).R = R;
IMM_Filter_Parameters(2,1).T = T;

IMM_Filter_Parameters(2,2).H=  H;                     
IMM_Filter_Parameters(2,2).Q = Q_coeffs(2)^2*Q;
IMM_Filter_Parameters(2,2).R = R;
IMM_Filter_Parameters(2,2).T = T;

TransProb(:,:,2) = [0.9 0.1;
                    0.1 0.9];
% Estimate Initialization

StateEstimates = zeros(4,Model_Num,Step_Num);
CombinedStateEstimate = zeros(4,Step_Num);
StateEstimates(:,1,1) = x0_bar;
StateEstimates(:,2,1) = x0_bar;
CombinedStateEstimate(:,1) = x0_bar;

StateEstimatesCov = zeros(4,4,Model_Num,Step_Num);
StateEstimatesCov(:,:,1,1) = P0_bar;
StateEstimatesCov(:,:,2,1) = P0_bar;
CombinedStateEstimateCov = zeros(4,4,Step_Num);
CombinedStateEstimateCov(:,:,1) = P0_bar;
Mu = zeros(Model_Num,Step_Num);
Mu(:,1) = [0.5;0.5];

CI_Local_Trackers(1).IMM_Parameters = IMM_Filter_Parameters(1,:);
CI_Local_Trackers(1).StateEstimates = StateEstimates;
CI_Local_Trackers(1).CombinedStateEstimate = CombinedStateEstimate;
CI_Local_Trackers(1).StateEstimatesCov = StateEstimatesCov;
CI_Local_Trackers(1).CombinedStateEstimateCov = CombinedStateEstimateCov;
CI_Local_Trackers(1).Mu = Mu;

CI_Local_Trackers(2).IMM_Parameters = IMM_Filter_Parameters(2,:);
CI_Local_Trackers(2).StateEstimates = StateEstimates;
CI_Local_Trackers(2).CombinedStateEstimate = CombinedStateEstimate;
CI_Local_Trackers(2).StateEstimatesCov = StateEstimatesCov;
CI_Local_Trackers(2).CombinedStateEstimateCov = CombinedStateEstimateCov;
CI_Local_Trackers(2).Mu = Mu;

StatePredictions = zeros(4,Model_Num);
StatePredictionsCov = zeros(4,4,Model_Num);
OutputPredictions = zeros(2,Model_Num);
OutputPredictionsCov = zeros(2,2,Model_Num);
KalmanGains = zeros(4,2,Model_Num);
L = zeros(1,Model_Num);


eps_covariance_intersection = zeros(MC_Num,Step_Num); % Normalized Estimation Error Squares Matrix
err_cov_int = zeros(4,Step_Num,MC_Num);

optim_param = 20;
w = linspace(1e-3,1-1e-3,optim_param);

for MC = 1:MC_Num
    
    for i=1:Step_Num-1
        if  i~=1 && mod(i,2) == 1 
                     
            for l=1:Model_Num
                
                [FusedCovarianceTrace,FusedModeProb,FusedModeCovariance,FusedModeEstimate] = SPCF(CI_Local_Trackers(2).Mu(l,i), CI_Local_Trackers(2).StateEstimates(:,l,i), ...
                                                                                                  CI_Local_Trackers(2).StateEstimatesCov(:,:,l,i), CI_Local_Trackers(1).Mu(:,i),...
                                                                                                  CI_Local_Trackers(1).StateEstimates(:,:,i), CI_Local_Trackers(1).StateEstimatesCov(:,:,:,i), w(1));
                
                minFusedCovarianceTrace = FusedCovarianceTrace;
                OptFusedModeProb = FusedModeProb;
                OptFusedModeCovariance = FusedModeCovariance;
                OptFusedModeEstimate = FusedModeEstimate;
                min_w = w(1);
                
                for m=2:optim_param
                
                [FusedCovarianceTrace,FusedModeProb,FusedModeCovariance,FusedModeEstimate] = SPCF(CI_Local_Trackers(2).Mu(l,i), CI_Local_Trackers(2).StateEstimates(:,l,i),...
                                                                                                  CI_Local_Trackers(2).StateEstimatesCov(:,:,l,i), CI_Local_Trackers(1).Mu(:,i),...
                                                                                                  CI_Local_Trackers(1).StateEstimates(:,:,i), CI_Local_Trackers(1).StateEstimatesCov(:,:,:,i), w(m));
                    if minFusedCovarianceTrace > FusedCovarianceTrace

                        minFusedCovarianceTrace = FusedCovarianceTrace;
                        OptFusedModeProb = FusedModeProb;
                        OptFusedModeCovariance = FusedModeCovariance;
                        OptFusedModeEstimate = FusedModeEstimate;
                        min_w = w(m);
                        
                    end
                end
     
                CI_Local_Trackers(2).StateEstimates(:,l,i) = OptFusedModeEstimate;
                
                CI_Local_Trackers(2).StateEstimatesCov(:,:,l,i) = OptFusedModeCovariance;
 
                CI_Local_Trackers(2).Mu(l,i) = OptFusedModeProb;
                
            end
            
            CI_Local_Trackers(2).Mu(:,i) = CI_Local_Trackers(2).Mu(:,i)/sum(CI_Local_Trackers(2).Mu(:,i));
            
            [CI_Local_Trackers(2).CombinedStateEstimate(:,i), CI_Local_Trackers(2).CombinedStateEstimateCov(:,:,i)] = immDurumBirlestirmeYap(CI_Local_Trackers(2).StateEstimates(:,:,i), CI_Local_Trackers(2).StateEstimatesCov(:,:,:,i),CI_Local_Trackers(2).Mu(:,i));

        end
        
        err_cov_int(:,i,MC) = x_k(:,i,MC)-CI_Local_Trackers(2).CombinedStateEstimate(:,i);
        eps_covariance_intersection(MC,i) = err_cov_int(:,i,MC)'*inv(CI_Local_Trackers(2).CombinedStateEstimateCov(:,:,i))*err_cov_int(:,i,MC);
        
        for k=1:2
            
            % Karistirma olasiliklari hesaplaniyor:
            %----------------------------------------
            [~, c, Mij] = immKaristirmaOlasiligiHesapla(TransProb(:,:,k), T, CI_Local_Trackers(k).Mu(:,i));
            
            % Karistirma yapiliyor:
            %---------------------------
            [MixedStateEstimates,MixedStateEstimatesCov] = immDurumKaristirmaYap(CI_Local_Trackers(k).StateEstimates(:,:,i), CI_Local_Trackers(k).StateEstimatesCov(:,:,:,i), Mij);
            
            for l=1:Model_Num
                
                % Prediction:
                
                [StatePredictions(:,l), ...
                    StatePredictionsCov(:,:,l), ...
                    OutputPredictions(:,l), ...
                    OutputPredictionsCov(:,:,l), ...
                    KalmanGains(:,:,l)] = kf_pre(MixedStateEstimates(:,l),...
                    MixedStateEstimatesCov(:,:,l),...
                    CI_Local_Trackers(k).IMM_Parameters(l));
                
                % Estimation:
                [...
                    CI_Local_Trackers(k).StateEstimates(:,l,i+1), ...
                    CI_Local_Trackers(k).StateEstimatesCov(:,:,l,i+1),...
                    L(l)] = kf_est(...
                    StatePredictions(:,l),...
                    StatePredictionsCov(:,:,l),...
                    OutputPredictions(:,l), ...
                    OutputPredictionsCov(:,:,l), ...
                    KalmanGains(:,:,l),...
                    yk_centralized(2*k-1:2*k,i+1,MC));
            end
            
            CI_Local_Trackers(k).Mu(:,i+1) = immModelOlasiliginiGuncelle(L, c);
            
            [CI_Local_Trackers(k).CombinedStateEstimate(:,i+1),CI_Local_Trackers(k).CombinedStateEstimateCov(:,:,i+1)] = immDurumBirlestirmeYap(CI_Local_Trackers(k).StateEstimates(:,:,i+1), CI_Local_Trackers(k).StateEstimatesCov(:,:,:,i+1),CI_Local_Trackers(k).Mu(:,i+1));
            
        end
    end
    
    err_cov_int(:,end,MC) = x_k(:,end,MC)-CI_Local_Trackers(2).StateEstimates(:,end);
    eps_covariance_intersection(MC,end) = err_cov_int(:,end,MC)'*inv(CI_Local_Trackers(2).StateEstimatesCov(:,:,end))*err_cov_int(:,end,MC);
end

%%

time = 0:Step_Num-1*T;
err_cent_mean = sqrt(mean(err_cent.^2,3));
err_decent_mean = sqrt(mean(err_decent.^2,3));
err_cov_int_mean = sqrt(mean(err_cov_int.^2,3));

eps_centralized_mean = mean(eps_centralized,1);
eps_decentralized_mean = mean(eps_decentralized,1);
eps_covariance_intersection_mean = mean(eps_covariance_intersection,1);

figure;
thresh_min = chi2inv(0.005,MC_Num*4)/MC_Num;
thresh_max = chi2inv(0.995,MC_Num*4)/MC_Num;
hold on
plot(time,eps_centralized_mean)
plot(time,eps_decentralized_mean)
plot(time,eps_covariance_intersection_mean)
plot(time,repmat(thresh_min,1,100))
plot(time,repmat(thresh_max,1,100))
legend({'Centralized', 'Decentralized','Covariance Intersection','Threshold Min','Threshold Max'}, 'fontsize', 10);
ylabel('NEES', 'fontsize', 12); grid on; xlabel('Time', 'fontsize', 12); grid on;

figure;
hold on;
plot(time(2:end),err_cent_mean(1,2:end));
plot(time(2:end),err_decent_mean(1,2:end));
plot(time(2:end),err_cov_int_mean(1,2:end));
legend({'Centralized', 'Decentralized','Covariance Intersection'}, 'fontsize', 10);
ylabel('RMS of Position X', 'fontsize', 12); grid on; xlabel('Time', 'fontsize', 12); grid on;

figure;
hold on;
plot(time(2:end),err_cent_mean(2,2:end));
plot(time(2:end),err_decent_mean(2,2:end));
plot(time(2:end),err_cov_int_mean(2,2:end));
legend({'Centralized', 'Decentralized','Covariance Intersection'}, 'fontsize', 10);
ylabel('RMS of Position Y', 'fontsize', 12); grid on; xlabel('Time', 'fontsize', 12); grid on;


figure;
hold on;
plot(time(2:end),err_cent_mean(3,2:end));
plot(time(2:end),err_decent_mean(3,2:end));
plot(time(2:end),err_cov_int_mean(3,2:end));
legend({'Centralized', 'Decentralized','Covariance Intersection'}, 'fontsize', 10);
ylabel('RMS of Velocity X', 'fontsize', 12); grid on; xlabel('Time', 'fontsize', 12); grid on;


figure;
hold on;
plot(time(2:end),err_cent_mean(4,2:end));
plot(time(2:end),err_decent_mean(4,2:end));
plot(time(2:end),err_cov_int_mean(4,2:end));
legend({'Centralized', 'Decentralized','Covariance Intersection'}, 'fontsize', 10);
ylabel('RMS of Velocity Y', 'fontsize', 12); grid on; xlabel('Time', 'fontsize', 12); grid on;