function infer_zim()
%%inference without delay

close all

load Pop
load zimdaily
Mzimdaily = movmean(zimdaily,7,1);  %%smoothe the data


% %%%%%%%%%%%%%%%
num_ens=300;
num_times=size(Mzimdaily,1);
obs_truth=Mzimdaily'; %%real data
lambda=1.015;%inflation factor
num_loc=size(Pop,1);%number of locations
N=Pop; %population
% %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%#ADD EAKF
%%%%set OEV which is required by EAKF. OEV is used by the EAKF to weigh the observations
%%%%# and model prior estimation of covid incidence, and to produce the model posterior estitmation 
% %%%#of both the state variables and parameters

OEV=zeros(num_loc,num_times);
for l=1:num_loc
    for t=1:num_times
          OEV(l,t)=max(4,obs_truth(l,t)^2/4); 
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%# start EAKF of network model
%%%%%initialization

x=initialize_eakf(Pop,num_ens);
x =checkbound(x,Pop);


%%%%run for one day to initialize system

for t=130:131 %1 day
    [x,obs_ens]=model_eakf(x,t,Pop);
end  

%Begin looping through observations
num_var=size(x,1); %number of state variables
xprior=NaN(num_var,num_ens,num_times);%prior
xpost=xprior;


 
 %for tt=101:num_times
 for tt=131:200

    %%% inflation of x before assimilation to avoid ensemble collapse
    x([1:50,54:73],:)=mean(x([1:50,54:73],:),2)*ones(1,num_ens)+lambda*(x([1:50,54:73],:)-mean(x([1:50,54:73],:),2)*ones(1,num_ens));

   % x=mean(x,2)*ones(1,num_ens)+lambda*(x-mean(x,2)*ones(1,num_ens));

    x =checkbound(x,Pop);

    %%%  Integrate forward one time step
        [x,obs_ens]=model_eakf(x,tt,Pop);
           

    
    xprior(:,:,tt)=x;%set prior
   
            
     %%loop through local observations

   for l=1:num_loc
            %Get the variance of the ensemble
            obs_var = OEV(l,tt);
            prior_var = var(obs_ens(l,:));
            post_var = prior_var*obs_var/(prior_var+obs_var);
                if prior_var==0%if degenerate
                    post_var=1e-3;
                    prior_var=1e-3;
                end
            prior_mean = mean(obs_ens(l,:));
            post_mean = post_var*(prior_mean/prior_var + obs_truth(l,tt)/obs_var);


%%%% Compute alphaa and adjust distribution to conform to posterior moments
        alpha = (obs_var/(obs_var+prior_var)).^0.5;
        dy = post_mean + alpha*(obs_ens(l,:)-prior_mean)-obs_ens(l,:);%%#observation adjustment



       %%%%%%%%%%%%%%%%%%%%%%%%%%%%Loop over each state variable
       rr=zeros(1,num_var);
        %%% SOME INDEX OF THE OBSERVATION/PROVINCE NUMBER jz 
    for idx=l
        for jz=1:5
            A=cov(x(5*(idx-1)+jz,:),obs_ens(l,:)); %%# covariance btwn each unobserved variabl/param nd observations calculated 4rm ensemble
            rr(5*(idx-1)+jz)=A(2,1)/prior_var;
        end
    end

        
       %%%%ADJUST PARAMETERS GLOBALLY except \alpha & beta 51-53
    
            for jw=num_loc*5+1:num_loc*5+3 %%%51:53 mu,Z,D
                A=cov(x(jw,:),obs_ens(l,:)); %%# covariance btwn each unobserved variabl/param nd observations calculated 4rm ensemble
                rr(jw)=A(2,1)/prior_var;
            end


            %%%%ADJUST \BETA LOCALLY  54-63

            for idx=l
                A=cov(x(num_loc*5+3+idx,:),obs_ens(l,:));   %%# covariance btwn each unobserved variabl/param nd observations calculated 4rm ensemble
                rr(num_loc*5+3+idx)=A(2,1)/prior_var;
            end


        %%%%ADJUST \ALPHA LOCALLY 64-73

            for idx=l
                A=cov(x(num_loc*5+13+idx,:),obs_ens(l,:));   %%# covariance btwn each unobserved variabl/param nd observations calculated 4rm ensemble
                rr(num_loc*5+13+idx)=A(2,1)/prior_var;
            end


    % end
        %%Get the adjusted variable 
        %%%# adjustments of unobserved variables/params r determined
        %%%bymultiplying the covariance rr with observation adjustment dy
        dx=rr'*dy;

        vardx(:,:,l)=dx(1:50,:);
        pramGLOBAL_dx(:,:,l)=dx(51:53,:);
        pramLOCAL_beta_dx(:,:,l)=dx(54:63,:);
        pramLOCAL_alpha_dx(:,:,l)=dx(64:end,:);


        sum_vardx=sum(vardx,3);
        sum_pramLOCAL_beta_dx =sum(pramLOCAL_beta_dx,3);
        sum_pramLOCAL_alpha_dx =sum(pramLOCAL_alpha_dx,3);
        sum_pramGLOBAL_dx =sum(pramGLOBAL_dx,3);

        avg_pramGLOBAL_dx=sum_pramGLOBAL_dx/num_loc;

        dx2 = [sum_vardx;avg_pramGLOBAL_dx;sum_pramLOCAL_beta_dx;sum_pramLOCAL_alpha_dx];
               
        obs_ens(l,:)=obs_ens(l,:)+dy;
        obs_ens(l,obs_ens(l,:)<0)=0;
    
   end

         x = x + dx2;
        %Corrections to DA produced aphysicalities
        x = checkbound(x,Pop);
        xnew = x;
        xpost(:,:,tt)=xnew;
        
       
 end

%%%%# Plot 



um=shiftdim(xpost,1);
xpost_mean=squeeze(mean(um));


tm=datetime(2021,7,10)+days(131:200);

%%%%SIMULATED Re
%%%bulawayo
r01=xpost(54,:,131:200).*xpost(53,:,131:200).*(xpost(64,:,131:200)+(1-xpost(64,:,131:200)).*xpost(51,:,131:200));
um=shiftdim(r01,1);
r01_mean=squeeze(mean(um));

%%%harare
r02=xpost(55,:,131:200).*xpost(53,:,131:200).*(xpost(65,:,131:200)+(1-xpost(65,:,131:200)).*xpost(51,:,131:200));
um=shiftdim(r02,1);
r02_mean=squeeze(mean(um));

%%%manicaland
r03=xpost(56,:,131:200).*xpost(53,:,131:200).*(xpost(66,:,131:200)+(1-xpost(66,:,131:200)).*xpost(51,:,131:200));
um=shiftdim(r03,1);
r03_mean=squeeze(mean(um));

%%%mashC
r04=xpost(57,:,131:200).*xpost(53,:,131:200).*(xpost(67,:,131:200)+(1-xpost(67,:,131:200)).*xpost(51,:,131:200));
um=shiftdim(r04,1);
r04_mean=squeeze(mean(um));

%%%mashE
r05=xpost(58,:,131:200).*xpost(53,:,131:200).*(xpost(68,:,131:200)+(1-xpost(68,:,131:200)).*xpost(51,:,131:200));
um=shiftdim(r05,1);
r05_mean=squeeze(mean(um));

%%%mashW
r06=xpost(59,:,131:200).*xpost(53,:,131:200).*(xpost(69,:,131:200)+(1-xpost(69,:,131:200)).*xpost(51,:,131:200));
um=shiftdim(r06,1);
r06_mean=squeeze(mean(um));

%%%mid
r07=xpost(60,:,131:200).*xpost(53,:,131:200).*(xpost(70,:,131:200)+(1-xpost(70,:,131:200)).*xpost(51,:,131:200));
um=shiftdim(r07,1);
r07_mean=squeeze(mean(um));

%%%masv
r08=xpost(61,:,131:200).*xpost(53,:,131:200).*(xpost(71,:,131:200)+(1-xpost(71,:,131:200)).*xpost(51,:,131:200));
um=shiftdim(r08,1);
r08_mean=squeeze(mean(um));

%%%matN
r09=xpost(62,:,131:200).*xpost(53,:,131:200).*(xpost(72,:,131:200)+(1-xpost(72,:,131:200)).*xpost(51,:,131:200));
um=shiftdim(r09,1);
r09_mean=squeeze(mean(um));

%%%matS
r010=xpost(63,:,131:200).*xpost(53,:,131:200).*(xpost(73,:,131:200)+(1-xpost(73,:,131:200)).*xpost(51,:,131:200));
um=shiftdim(r010,1);
r010_mean=squeeze(mean(um));



%%%%%%%%%%%%%%%


       %%percentiles byo
Pr01=r01(1,:,:);
Pr01_low=prctile(Pr01,2.5,2);
Pr01_high=prctile(Pr01,97.5,2);

%%harare
Pr02=r02(1,:,:);
Pr02_low=prctile(Pr02,2.5,2);
Pr02_high=prctile(Pr02,97.5,2);

%%mani
Pr03=r03(1,:,:);
Pr03_low=prctile(Pr03,2.5,2);
Pr03_high=prctile(Pr03,97.5,2);


%%mashC
Pr04=r04(1,:,:);
Pr04_low=prctile(Pr04,2.5,2);
Pr04_high=prctile(Pr04,97.5,2);


%%mashE
Pr05=r05(1,:,:);
Pr05_low=prctile(Pr05,2.5,2);
Pr05_high=prctile(Pr05,97.5,2);


%%mashW
Pr06=r06(1,:,:);
Pr06_low=prctile(Pr06,2.5,2);
Pr06_high=prctile(Pr06,97.5,2);


%%mid
Pr07=r07(1,:,:);
Pr07_low=prctile(Pr07,2.5,2);
Pr07_high=prctile(Pr07,97.5,2);


%%masv
Pr08=r08(1,:,:);
Pr08_low=prctile(Pr08,2.5,2);
Pr08_high=prctile(Pr08,97.5,2);


%%matN
Pr09=r09(1,:,:);
Pr09_low=prctile(Pr09,2.5,2);
Pr09_high=prctile(Pr09,97.5,2);


%%matS
Pr010=r010(1,:,:);
Pr010_low=prctile(Pr010,2.5,2);
Pr010_high=prctile(Pr010,97.5,2);

% % 
 %%%%%%%%%parameters and ensemble    
%%%%%%%%%%%%%%%%ALL
Ppara_all=xpost(:,:,131:200);
%Ppara_all=xpost(:,:,:);
Ppara_low=prctile(Ppara_all,2.5,2);
Ppara_high=prctile(Ppara_all,97.5,2);


xx=tm;

%%%%RE percentiles
%%%byo

yy1=squeeze(Pr01_low(1,1,:))';
yy2=squeeze(Pr01_high(1,1,:))';

%%%hre
yy1002=squeeze(Pr02_low(1,1,:))';
yy2002=squeeze(Pr02_high(1,1,:))';

%%%mani
yy1003=squeeze(Pr03_low(1,1,:))';
yy2003=squeeze(Pr03_high(1,1,:))';

%%%mashC
yy1004=squeeze(Pr04_low(1,1,:))';
yy2004=squeeze(Pr04_high(1,1,:))';

%%%mashE
yy1005=squeeze(Pr05_low(1,1,:))';
yy2005=squeeze(Pr05_high(1,1,:))';

%%%mashW
yy1006=squeeze(Pr06_low(1,1,:))';
yy2006=squeeze(Pr06_high(1,1,:))';

%%%mid
yy1007=squeeze(Pr07_low(1,1,:))';
yy2007=squeeze(Pr07_high(1,1,:))';

%%%masv
yy1008=squeeze(Pr08_low(1,1,:))';
yy2008=squeeze(Pr08_high(1,1,:))';

%%%matN
yy1009=squeeze(Pr09_low(1,1,:))';
yy2009=squeeze(Pr09_high(1,1,:))';

%%%matS
yy10010=squeeze(Pr010_low(1,1,:))';
yy20010=squeeze(Pr010_high(1,1,:))';




%%%%%beta pcentiles
yy12=squeeze(Ppara_low(54,1,:))';
yy22=squeeze(Ppara_high(54,1,:))';

yy13=squeeze(Ppara_low(55,1,:))';
yy23=squeeze(Ppara_high(55,1,:))';

yy14=squeeze(Ppara_low(56,1,:))';
yy24=squeeze(Ppara_high(56,1,:))';

yy15=squeeze(Ppara_low(57,1,:))';
yy25=squeeze(Ppara_high(57,1,:))';

yy16=squeeze(Ppara_low(58,1,:))';
yy26=squeeze(Ppara_high(58,1,:))';

yy17=squeeze(Ppara_low(59,1,:))';
yy27=squeeze(Ppara_high(59,1,:))';

yy18=squeeze(Ppara_low(60,1,:))';
yy28=squeeze(Ppara_high(60,1,:))';

yy19=squeeze(Ppara_low(61,1,:))';
yy29=squeeze(Ppara_high(61,1,:))';

yy110=squeeze(Ppara_low(62,1,:))';
yy210=squeeze(Ppara_high(62,1,:))';

yy111=squeeze(Ppara_low(63,1,:))';
yy211=squeeze(Ppara_high(63,1,:))';


%%%%%%Alpha percentiles
%%
yy113=squeeze(Ppara_low(64,1,:))';
yy213=squeeze(Ppara_high(64,1,:))';

yy114=squeeze(Ppara_low(65,1,:))';
yy214=squeeze(Ppara_high(65,1,:))';


yy115=squeeze(Ppara_low(66,1,:))';
yy215=squeeze(Ppara_high(66,1,:))';


yy116=squeeze(Ppara_low(67,1,:))';
yy216=squeeze(Ppara_high(67,1,:))';


yy117=squeeze(Ppara_low(68,1,:))';
yy217=squeeze(Ppara_high(68,1,:))';


yy118=squeeze(Ppara_low(69,1,:))';
yy218=squeeze(Ppara_high(69,1,:))';


yy119=squeeze(Ppara_low(70,1,:))';
yy219=squeeze(Ppara_high(70,1,:))';


yy1120=squeeze(Ppara_low(71,1,:))';
yy2120=squeeze(Ppara_high(71,1,:))';


yy1121=squeeze(Ppara_low(72,1,:))';
yy2121=squeeze(Ppara_high(72,1,:))';


yy1122=squeeze(Ppara_low(73,1,:))';
yy2122=squeeze(Ppara_high(73,1,:))';




%%%%%parameter time series


figure(31)
        subplot(3,1,1)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy12, fliplr(yy22)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,xpost_mean(131:200,54),'r','Linewidth',2)% location 1 
        title('\beta_1 Bulawayo')
        legend('95% CrI ','posterior mean','Location','Best')
        legend boxoff 
        set (gca,'FontSize',12)

        subplot(3,1,2)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy113, fliplr(yy213)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,xpost_mean(131:200,64),'r','Linewidth',2)% location 1 
        ylabel('\alpha prior ', 'FontSize', 11)
        title('\alpha_1')
        set (gca,'FontSize',12)
        
        subplot(3,1,3)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy1, fliplr(yy2)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,r01_mean(:,:),'r','Linewidth',2)% location 1 
        xlabel('Time (days)', 'FontSize', 10)
        ylabel('R_t', 'FontSize', 11)
        set (gca,'FontSize',12)


        figure(32)
        subplot(3,1,1)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy13, fliplr(yy23)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,xpost_mean(131:200,55),'r','Linewidth',2)% location 1 
        ylabel('\beta prior', 'FontSize', 11)
        title('\beta_2 Harare')
        legend('95% CrI ','posterior mean','Location','Best')
        legend boxoff 
        set (gca,'FontSize',12)

        subplot(3,1,2)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy114, fliplr(yy214)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,xpost_mean(131:200,65),'r','Linewidth',2)% location 1 
        ylabel('\alpha prior ', 'FontSize', 11)
        title('\alpha_2')
        set (gca,'FontSize',12)
        
        subplot(3,1,3)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy1002, fliplr(yy2002)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,r02_mean(:,:),'r','Linewidth',2)% location 1 
        xlabel('Time (days)', 'FontSize', 10)
        ylabel('R_t', 'FontSize', 11)
        set (gca,'FontSize',12)


        figure(33)
        subplot(3,1,1)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy14, fliplr(yy24)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,xpost_mean(131:200,56),'r','Linewidth',2)% location 1 
        ylabel('\beta prior', 'FontSize', 11)
        title('\beta_3 Manicaland')
        legend('95% CrI ','posterior mean','Location','Best')
        legend boxoff 
        set (gca,'FontSize',12)

        subplot(3,1,2)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy115, fliplr(yy215)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,xpost_mean(131:200,66),'r','Linewidth',2)% location 1 
        ylabel('\alpha prior ', 'FontSize', 11)
        title('\alpha_3')
        set (gca,'FontSize',12)
        
        subplot(3,1,3)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy1003, fliplr(yy2003)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,r03_mean(:,:),'r','Linewidth',2)% location 1 
        xlabel('Time (days)', 'FontSize', 10)
        ylabel('R_t', 'FontSize', 11)
        set (gca,'FontSize',12)


        figure(34)
        subplot(3,1,1)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy15, fliplr(yy25)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,xpost_mean(131:200,57),'r','Linewidth',2)% location 1 
        ylabel('\beta prior', 'FontSize', 11)
        title('\beta_4 Mashonaland Central')
        legend('95% CrI ','posterior mean','Location','Best')
        legend boxoff 
        set (gca,'FontSize',12)

        subplot(3,1,2)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy116, fliplr(yy216)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,xpost_mean(131:200,67),'r','Linewidth',2)% location 1 
        ylabel('\alpha prior ', 'FontSize', 11)
        title('\alpha_4')
        set (gca,'FontSize',12)
        
        subplot(3,1,3)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy1004, fliplr(yy2004)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,r04_mean(:,:),'r','Linewidth',2)% location 1 
        xlabel('Time (days)', 'FontSize', 10)
        ylabel('R_t', 'FontSize', 11)
        set (gca,'FontSize',12)


        figure(35)
        subplot(3,1,1)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy16, fliplr(yy26)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,xpost_mean(131:200,58),'r','Linewidth',2)% location 1 
        ylabel('\beta prior', 'FontSize', 11)
        title('\beta_5 Mashonaland East')
        legend('95% CrI ','posterior mean','Location','Best')
        legend boxoff 
        set (gca,'FontSize',12)

        subplot(3,1,2)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy117, fliplr(yy217)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,xpost_mean(131:200,68),'r','Linewidth',2)% location 1 
        ylabel('\alpha prior ', 'FontSize', 11)
        title('\alpha_5')
        set (gca,'FontSize',12)
        
        subplot(3,1,3)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy1005, fliplr(yy2005)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,r05_mean(:,:),'r','Linewidth',2)% location 1 
        xlabel('Time (days)', 'FontSize', 10)
        ylabel('R_t', 'FontSize', 11)
        set (gca,'FontSize',12)


        figure(36)
        subplot(3,1,1)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy17, fliplr(yy27)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,xpost_mean(131:200,59),'r','Linewidth',2)% location 1 
        ylabel('\beta prior', 'FontSize', 11)
        title('\beta_6 Mashonaland West')
        legend('95% CrI ','posterior mean','Location','Best')
        legend boxoff 
        set (gca,'FontSize',12)

        subplot(3,1,2)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy118, fliplr(yy218)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,xpost_mean(131:200,69),'r','Linewidth',2)% location 1 
        ylabel('\alpha prior ', 'FontSize', 11)
        title('\alpha_6')
        set (gca,'FontSize',12)
        
        subplot(3,1,3)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy1006, fliplr(yy2006)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,r06_mean(:,:),'r','Linewidth',2)% location 1 
        xlabel('Time (days)', 'FontSize', 10)
        ylabel('R_t', 'FontSize', 11)
        set (gca,'FontSize',12)


        figure(37)
        subplot(3,1,1)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy18, fliplr(yy28)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,xpost_mean(131:200,60),'r','Linewidth',2)% location 1 
        ylabel('\beta prior', 'FontSize', 11)
        title('\beta_7 Midlands')
        legend('95% CrI ','posterior mean','Location','Best')
        legend boxoff 
        set (gca,'FontSize',12)

        subplot(3,1,2)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy119, fliplr(yy219)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,xpost_mean(131:200,70),'r','Linewidth',2)% location 1 
        ylabel('\alpha prior ', 'FontSize', 11)
        title('\alpha_7')
        set (gca,'FontSize',12)
        
        subplot(3,1,3)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy1007, fliplr(yy2007)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,r07_mean(:,:),'r','Linewidth',2)% location 1 
        xlabel('Time (days)', 'FontSize', 10)
        ylabel('R_t', 'FontSize', 11)
        set (gca,'FontSize',12)


        figure(38)
        subplot(3,1,1)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy19, fliplr(yy29)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,xpost_mean(131:200,61),'r','Linewidth',2)% location 1 
        ylabel('\beta prior', 'FontSize', 11)
        title('\beta_8 Masvingo')
        legend('95% CrI ','posterior mean','Location','Best')
        legend boxoff 
        set (gca,'FontSize',12)

        subplot(3,1,2)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy1120, fliplr(yy2120)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,xpost_mean(131:200,71),'r','Linewidth',2)% location 1 
        ylabel('\alpha prior ', 'FontSize', 11)
        title('\alpha_8')
        set (gca,'FontSize',12)
        
        subplot(3,1,3)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy1008, fliplr(yy2008)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,r08_mean(:,:),'r','Linewidth',2)% location 1 
        xlabel('Time (days)', 'FontSize', 10)
        ylabel('R_t', 'FontSize', 11)
        set (gca,'FontSize',12)


        figure(39)
        subplot(3,1,1)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy110, fliplr(yy210)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,xpost_mean(131:200,62),'r','Linewidth',2)% location 1 
        ylabel('\beta prior', 'FontSize', 11)
        title('\beta_9 Matebeleland North')
        legend('95% CrI ','posterior mean','Location','Best')
        legend boxoff 
        set (gca,'FontSize',12)

        subplot(3,1,2)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy1121, fliplr(yy2121)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,xpost_mean(131:200,72),'r','Linewidth',2)% location 1 
        ylabel('\alpha prior ', 'FontSize', 11)
        title('\alpha_9')
        set (gca,'FontSize',12)
        
        subplot(3,1,3)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy1009, fliplr(yy2009)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,r09_mean(:,:),'r','Linewidth',2)% location 1 
        xlabel('Time (days)', 'FontSize', 10)
        ylabel('R_t', 'FontSize', 11)
        set (gca,'FontSize',12)


        figure(40)
        subplot(3,1,1)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy111, fliplr(yy211)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,xpost_mean(131:200,63),'r','Linewidth',2)% location 1 
        ylabel('\beta prior', 'FontSize', 11)
        title('\beta_{10} Matebeleland South')
        legend('95% CrI ','posterior mean','Location','Best')
        legend boxoff 
        set (gca,'FontSize',12)

        subplot(3,1,2)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy1122, fliplr(yy2122)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,xpost_mean(131:200,73),'r','Linewidth',2)% location 1 
        ylabel('\alpha prior ', 'FontSize', 11)
        title('\alpha_{10}')
        set (gca,'FontSize',12)
        
        subplot(3,1,3)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy10010, fliplr(yy20010)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,r010_mean(:,:),'r','Linewidth',2)% location 1 
        xlabel('Time (days)', 'FontSize', 10)
        ylabel('R_t', 'FontSize', 11)
        set (gca,'FontSize',12)



%%%%observations

yy31=squeeze(Ppara_low(5,1,:))';
yy41=squeeze(Ppara_high(5,1,:))';

yy321=squeeze(Ppara_low(10,1,:))';
yy421=squeeze(Ppara_high(10,1,:))';

yy331=squeeze(Ppara_low(15,1,:))';
yy431=squeeze(Ppara_high(15,1,:))';

yy341=squeeze(Ppara_low(20,1,:))';
yy441=squeeze(Ppara_high(20,1,:))';

yy351=squeeze(Ppara_low(25,1,:))';
yy451=squeeze(Ppara_high(25,1,:))';

yy361=squeeze(Ppara_low(30,1,:))';
yy461=squeeze(Ppara_high(30,1,:))';

yy371=squeeze(Ppara_low(35,1,:))';
yy471=squeeze(Ppara_high(35,1,:))';

yy381=squeeze(Ppara_low(40,1,:))';
yy481=squeeze(Ppara_high(40,1,:))';

yy391=squeeze(Ppara_low(45,1,:))';
yy491=squeeze(Ppara_high(45,1,:))';

yy301=squeeze(Ppara_low(50,1,:))';
yy401=squeeze(Ppara_high(50,1,:))';

data=obs_truth(:,131:200);



figure (91)

x2 = [xx, fliplr(xx)];
inBetween = [yy31, fliplr(yy41)];
fill(x2, inBetween, 'r','FaceAlpha',0.3,'EdgeColor', 'none');
hold on
plot(tm,xpost_mean(131:200,5),'r-','Linewidth', 1.5)
hold on
plot(tm, data(1,:), "b.")
 xlabel('Date', 'FontSize', 12.5)
 ylabel('Daily new cases (population)', 'FontSize', 12.5)
 title('Bulawayo ','FontWeight','bold')
 legend('95% CrI','New cases posterior mean','Data', 'FontSize', 10,'Location','Best')
 legend boxoff 
 set (gca,'FontSize',16)

 figure (92)
 x2 = [xx, fliplr(xx)];
 inBetween = [yy321, fliplr(yy421)];
 fill(x2, inBetween, 'r','FaceAlpha',0.3,'EdgeColor', 'none');
 hold on
 plot(tm,xpost_mean(131:200,10),'r-','Linewidth', 1.5)
 hold on
 plot(tm, data(2,:), "b.")
 xlabel('Date', 'FontSize', 12.5)
 ylabel('Daily new cases (population)', 'FontSize', 12.5)
 title('Harare ','FontWeight','bold')
 legend('95% CrI','New cases posterior mean','Data', 'FontSize', 10,'Location','Best')
 legend boxoff 
 set (gca,'FontSize',16)

 figure (93)

 x2 = [xx, fliplr(xx)];
 inBetween = [yy331, fliplr(yy431)];
 fill(x2, inBetween, 'r','FaceAlpha',0.3,'EdgeColor', 'none');
 hold on
 plot(tm,xpost_mean(131:200,15),'r-','Linewidth', 1.5)
 hold on
 plot(tm, data(3,:), "b.")
 xlabel('Date', 'FontSize', 12.5)
 ylabel('Daily new cases (population)', 'FontSize', 12.5)
 title('Manicaland ','FontWeight','bold')
 legend('95% CrI','New cases posterior mean','Data', 'FontSize', 10,'Location','Best')
 legend boxoff 
 set (gca,'FontSize',16) 

 figure (94)

 x2 = [xx, fliplr(xx)];
 inBetween = [yy341, fliplr(yy441)];
 fill(x2, inBetween, 'r','FaceAlpha',0.3,'EdgeColor', 'none');
 hold on
 plot(tm,xpost_mean(131:200,20),'r-','Linewidth', 1.5)
 hold on
 plot(tm, data(4,:), "b.")
 xlabel('Date', 'FontSize', 12.5)
 ylabel('Daily new cases (population)', 'FontSize', 12.5)
 title('Mashonaland Central ','FontWeight','bold')
 legend('95% CrI','New cases posterior mean','Data', 'FontSize', 10,'Location','Best')
 legend boxoff 
 set (gca,'FontSize',16)


figure (95)

 x2 = [xx, fliplr(xx)];
 inBetween = [yy351, fliplr(yy451)];
 fill(x2, inBetween, 'r','FaceAlpha',0.3,'EdgeColor', 'none');
 hold on
 plot(tm,xpost_mean(131:200,25),'r-','Linewidth', 1.5)
 hold on
 plot(tm, data(5,:), "b.")
 xlabel('Date', 'FontSize', 12.5)
 ylabel('Daily new cases (population)', 'FontSize', 12.5)
 title('Mashonaland East ','FontWeight','bold')
 legend('95% CrI','New cases posterior mean','Data', 'FontSize', 10,'Location','Best')
 legend boxoff 
 set (gca,'FontSize',16) 

figure (96)
 x2 = [xx, fliplr(xx)];
 inBetween = [yy361, fliplr(yy461)];
 fill(x2, inBetween, 'r','FaceAlpha',0.3,'EdgeColor', 'none');
 hold on
 plot(tm,xpost_mean(131:200,30),'r-','Linewidth', 1.5)
 hold on
 plot(tm, data(6,:), "b.")
 xlabel('Date', 'FontSize', 12.5)
 ylabel('Daily new cases (population)', 'FontSize', 12.5)
 title('Mashonaland West ','FontWeight','bold')
 legend('95% CrI','New cases posterior mean','Data', 'FontSize', 10,'Location','Best')
 legend boxoff 
 set (gca,'FontSize',16) 


 figure (97)
 x2 = [xx, fliplr(xx)];
 inBetween = [yy371, fliplr(yy471)];
 fill(x2, inBetween, 'r','FaceAlpha',0.3,'EdgeColor', 'none');
 hold on
 plot(tm,xpost_mean(131:200,35),'r-','Linewidth', 1.5)
 hold on
 plot(tm, data(7,:), "b.")
 xlabel('Date', 'FontSize', 12.5)
 ylabel('Daily new cases (population)', 'FontSize', 12.5)
 title('Midlands','FontWeight','bold')
 legend('95% CrI','New cases posterior mean','Data', 'FontSize', 10,'Location','Best')
 legend boxoff 
 set (gca,'FontSize',16)

 figure (98)
 x2 = [xx, fliplr(xx)];
 inBetween = [yy381, fliplr(yy481)];
 fill(x2, inBetween, 'r','FaceAlpha',0.3,'EdgeColor', 'none');
 hold on
 plot(tm,xpost_mean(131:200,40),'r-','Linewidth', 1.5)
 hold on
 plot(tm, data(8,:), "b.")
 xlabel('Date', 'FontSize', 12.5)
 ylabel('Daily new cases (population)', 'FontSize', 12.5)
 title('Masvingo ','FontWeight','bold')
 legend('95% CrI','New cases posterior mean','Data', 'FontSize', 10,'Location','Best')
 legend boxoff 
 set (gca,'FontSize',16)

 figure (99)
 x2 = [xx, fliplr(xx)];
 inBetween = [yy391, fliplr(yy491)];
 fill(x2, inBetween, 'r','FaceAlpha',0.3,'EdgeColor', 'none');
 hold on
 plot(tm,xpost_mean(131:200,45),'r-','Linewidth', 1.5)
 hold on
 plot(tm, data(9,:), "b.")
 xlabel('Date', 'FontSize', 12.5)
 ylabel('Daily new cases (population)', 'FontSize', 12.5)
 title('Matebeland North ','FontWeight','bold')
 legend('95% CrI','New cases posterior mean','Data', 'FontSize', 10,'Location','Best')
 legend boxoff 
 set (gca,'FontSize',16)

 figure (100)

 x2 = [xx, fliplr(xx)];
 inBetween = [yy301, fliplr(yy401)];
 fill(x2, inBetween, 'r','FaceAlpha',0.3,'EdgeColor', 'none');
 hold on
 plot(tm,xpost_mean(131:200,50),'r-','Linewidth', 1.5)
 hold on
plot(tm, data(10,:), "b.")
 xlabel('Date', 'FontSize', 12.5)
 ylabel('Daily new cases (population)', 'FontSize', 12.5)
 title('Matebeland South ','FontWeight','bold')
 legend('95% CrI','New cases posterior mean','Data', 'FontSize', 10,'Location','Best')
 legend boxoff 
 set (gca,'FontSize',16)







%%%%%%%%%%%%%%%#CHECKBOUND inference


function x=checkbound(x,Pop)

num_loc=size(Pop,1);
N=Pop;

 %%%parameters
betal=0.5;betau=2;%transmission rate 
mul=0.8;muu=0.8;%relative transmissibility
Zl=4; Zu=4;
alphal=0.01;alphau=0.1;
Dl=4; Du=4; %infectious period D=3/2
% 
xmin=[mul,Zl,Dl,ones(1,num_loc)*betal,ones(1,num_loc)*alphal];
xmax=[muu,Zu,Du,ones(1,num_loc)*betau,ones(1,num_loc)*alphau];



xmin(14)=0.04;
xmin(15)=0.03;
xmin(16)=0.02;
xmin(17)=0.01;
xmin(18)=0.03;
xmin(19)=0.02;
xmin(20)=0.01;
xmin(21)=0.02;
xmin(22)=0.04;
xmin(23)=0.03;


%%%redistribute out of bound parmaeter members
for i=1:23
    temp=x(end-23+i,:);
    index=(temp<xmin(i))|(temp>xmax(i));
    index_out=find(index>0);
    index_in=find(index==0);
    %redistribute out bound ensemble members
    x(end-23+i,index_out)=datasample(x(end-23+i,index_in),length(index_out));
end




%%state variables
for i=1:num_loc

     %S
  
    x((i-1)*5+1,x((i-1)*5+1,:)>N(i))=N(i)-1;

    %     %E
       x((i-1)*5+2,x((i-1)*5+2,:)>N(i))=N(i)-1;
    %Ir
        x((i-1)*5+3,x((i-1)*5+3,:)>N(i))=N(i)-1;
    %Iu
       x((i-1)*5+4,x((i-1)*5+4,:)>N(i))=N(i)-1;
    %obs
        x((i-1)*5+5,x((i-1)*5+5,:)>N(i))=N(i)-1;




end
% 
% %%%%redistribute out of bound state varaibles members
for i=1:num_loc
    for jj=1:5
    temp_var=x((i-1)*5+jj,:);
        index_var=(temp_var<0);
        index_var_out=find(index_var>0);
        index_var_in=find(index_var==0);
        %redistribute out bound ensemble members
    x((i-1)*5+jj,index_var_out)=datasample(x((i-1)*5+jj,index_var_in),length(index_var_out));
    end
end











