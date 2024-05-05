close; 
clear;
format long

%% parameters, input-dns, 
utau=1;
retau = 395;
input_dns=load('chan_395_dns.data'); % y, U, tke, eps

%% compare dns and k-ep results
figure;
plot(input_dns(:,1),input_dns(:,2)*utau,'ok')

%%
outInput=zeros(size(input_dns(:,1),1),4);
outInput(:,1)=input_dns(:,1);
outInput(:,2)=input_dns(:,2)*utau;
outInput(:,3)=input_dns(:,3)*utau^2;
% outInput(:,4)=input_dns(:,4)*1/retau./(utau^4)./(0.09*input_dns(:,3)*utau^2);

outInput(:,4)=input_dns(:,4)./(utau^4);

%%
figure;
plot(outInput(:,1), outInput(:,2),'-o',input_dns(:,1),input_dns(:,2),'-k')

figure;
plot(outInput(:,1), outInput(:,3),'-o',input_dns(:,1),input_dns(:,3),'-k')

figure;
plot(outInput(:,1), outInput(:,4),'-o')
 ylim([0 .4])
% save input.data.case0 outInput -ASCII
