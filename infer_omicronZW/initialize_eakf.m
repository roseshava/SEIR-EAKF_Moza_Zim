
function x=initialize_eakf(Pop,num_ens)


num_loc=size(Pop,1);
N=Pop;

Sl=0.15; Su=0.8; %lower, upper fraction
El=0; Eu=0.0001;
Irl=0; Iru=0;
Iul=0; Iuu=0;



% % % %estimate all parameters
betal=0.5;betau=2;%transmission rate 
mul=0.8;muu=0.8;%relative transmissibility
Zl=4; Zu=4;
alphal=0.01;alphau=0.1;
Dl=4; Du=4; %infectious period D=3/2


%%range of model state including variables and parameters
xmin=[];
xmax=[];
for i=1:num_loc
    xmin=[xmin,Sl*N(i),El*N(i),Irl*N(i),Iul*N(i),0];
    xmax=[xmax,Su*N(i),Eu*N(i),Iru*N(i),Iuu*N(i),0];

end

xmin=[xmin,mul,Zl,Dl,ones(1,num_loc)*betal,ones(1,num_loc)*alphal];
xmax=[xmax,muu,Zu,Du,ones(1,num_loc)*betau,ones(1,num_loc)*alphau];



xmin(64)=0.04;
xmin(65)=0.03;
xmin(66)=0.02;
xmin(67)=0.01;
xmin(68)=0.03;
xmin(69)=0.02;
xmin(70)=0.01;
xmin(71)=0.02;
xmin(72)=0.04;
xmin(73)=0.03;



%%Latin Hypercubic Sampling
x=lhsu(xmin,xmax,num_ens);
x=x';

for i=1:num_loc
    x((i-1)*5+1:(i-1)*5+4,:)=round(x((i-1)*5+1:(i-1)*5+4,:));
end




%% LHS from uniform distribution
function s=lhsu(xmin,xmax,nsample)

nvar=length(xmin);
ran=rand(nsample,nvar);
s=zeros(nsample,nvar);
for j=1: nvar
    idx=randperm(nsample);
   P =(idx'-ran(:,j))/nsample;
   s(:,j) = xmin(j) + P.* (xmax(j)-xmin(j));
end