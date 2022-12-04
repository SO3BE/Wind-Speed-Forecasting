clc
close all
clear all
 [chosenfile,chosendirectory] = uigetfile({'*.xlsx';'*.csv'});
   filePath = [chosendirectory chosenfile]
        X = readtable(filePath); 
         data.seriesdataHeder = X.Properties.VariableNames(1,:);
        X = table2array(X(:,:));
       
         [chosenfile,chosendirectory] = uigetfile({'*.xlsx';'*.csv'});
   filePath = [chosendirectory chosenfile]
        y = readtable(filePath);
         data.seriesdataHeder = y.Properties.VariableNames(1,:);
        Y= table2array(y(:,:));
       
      for i=1:16
          figure(i)
          e(:,i) = Y(:,i)-X(:,i);
            MSE(i)= mean(e(:,i).^2);
            RMSE(i)  = sqrt(MSE(i));
            NRMSE(i) = RMSE(i)/mean(Y(:,i));
            ErrorMean(i) = mean(e(:,i));
            ErrorStd(i)  = std(e(:,i));
            
x = Y(:,i);
y = X(:,i);

b1 = x\y;
yCalc1 = b1*x;
scatter(x,y,'MarkerEdgeColor',[0 0.4470 0.7410],'LineWidth',.7);
hold('on');
plot(x,yCalc1,'Color',[0.8500 0.3250 0.0980]);
xlabel('Prediction');
ylabel('Target');
grid minor
% xgrid = 'on';
% disp.YGrid = 'on';
p = [ones(length(x),1) x];
b = p\y;
yCalc2 = p*b;
plot(x,yCalc2,'-.','MarkerSize',4,"LineWidth",.1,'Color',[0.9290 0.6940 0.1250])
legend('Data','Fit','Y=T','Location','best');
%
Rsq2(i) = 1 -  sum((y - yCalc1).^2)/sum((y - mean(y)).^2);

 
    data.Err.RSqur_Tr = Rsq2(i);
    title(['Train Data, R^2 = ' num2str(Rsq2(i))]);
           

    
      end
  
      
      
      