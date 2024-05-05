close;  clear; format long
%% parameters, input-dns,
utau_rans=1;  % same as the input param. in c code

% only use y, the coordinate? from case0, force convection case results
input_org=load('input.data.case0');
% input dns
load('C:\unimelb_work\cited_data\duncanData\re395_ensemble.mat');
re395= re395_ensemble;
clear re395_ensemble

for i=7
    z=re395(i).data.z;
    utau=re395(i).parameters.utau;
    %  gbeta=re395(i).parameters.gbeta;
    nu=re395(i).parameters.nu;
    
    tke=(re395(i).data.uu+re395(i).data.vv+re395(i).data.ww)/2/mean(utau)^2;
    ubar=re395(i).data.u/mean(utau);
    num_points=size(z,1);
    eps =zeros(num_points,1);
    
    input_dns=[z, ubar, tke, eps];
end
%% cal, actural input Retau, mean(utau), but not each side
retau = mean(utau)*1./nu; % here, use global Re_\tau
disp(retau);
% %% compare casei vs case 0 results
 figure;
 plot(input_org(:,1),input_org(:,2),'-.x'), hold on
 plot(input_dns(:,1),input_dns(:,2)*utau_rans,'ok')

%%
outInput=zeros(size(input_org(:,1),1),4);
outInput(:,1)=input_org(:,1);
outInput(:,2)=interp1(input_dns(:,1),input_dns(:,2)*utau_rans,input_org(:,1), 'pchip');
outInput(:,3)=interp1(input_dns(:,1),input_dns(:,3)*utau_rans^2,input_org(:,1), 'pchip');
outInput(:,4)=input_org(:,4); % eps

% %%
figure;
plot(outInput(:,1), outInput(:,2),'-o',input_dns(:,1),input_dns(:,2)*utau_rans,'-k')
figure;
plot(outInput(:,1), outInput(:,3),'-o',input_dns(:,1),input_dns(:,3)*utau_rans^2,'-k')
figure;
plot(outInput(:,1), outInput(:,4),'-o', input_org(:,1),input_org(:,4),'-r')
ylim([0 1])

%%
% save input.data.case6 outInput -ASCII