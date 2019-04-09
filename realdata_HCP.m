
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%4/08/19
%real data example
%High Order PCA for brain networks

clear all;
close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%load HCP data - tensor networks and covariates
% data are produced by using parameters 20-240mm, desikan 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('data/HCP_200sub_data.mat');
% CAS_networs 68*68*200, 200 CSA networks from 200 HCP subjects (see Zhang
% et. al. 2018 NeuroImage for the definition of CSA network)

% ReadEng_AgeAdj - Test score of read English test, see HCP (https://db.humanconnectome.org) for more
% details.  
% 1-100 subjects have low scores
% 101-200 subjects have high scores
low_score = ReadEng_AgeAdj(1:100);
high_score = ReadEng_AgeAdj(101:200);


subN = 200;
Kcomp = 30;

selected_tensor = CSA_networks;

%remove the mean
mean_X = mean(selected_tensor,3);
for isub = 1:size(selected_tensor,3)
    X_c(:,:,isub) = selected_tensor(:,:,isub) - mean_X;
end
X_tensor = tensor(X_c);

%TN-PCA
[Vo,DiagDo,Do,Uo] = hooi_popNet(X_tensor,Kcomp);
options = struct('proj',1);
options.startv = Vo;
options.startu = Uo;
[V,D,U,W,Xh] = hopca_popNet(X_tensor,Kcomp,options);

newU = U;

low_score_U = newU(1:100,1:Kcomp);
high_score_U = newU(101:200,1:Kcomp);

%plot the first 3 PC scores;
figure(100);clf;
for i=1:size(low_score_U,1)
    hold on;
    scatter3(low_score_U(i,1),low_score_U(i,2),low_score_U(i,3),80,low_score(i),'fill')
end
for i=1:size(high_score_U,1)
    hold on;
    scatter3(high_score_U(i,1),high_score_U(i,2),high_score_U(i,3),80,high_score(i),'fill')
end
set(gca,'fontsize',26)
colormap(parula);
grid on;