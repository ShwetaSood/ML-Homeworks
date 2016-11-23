clear all;
clc;

%Creating training data
tmp_data=importdata('segmentation.data.txt',',');
training=tmp_data.textdata(4:size(tmp_data.textdata,1));
arr=strsplit(cell2mat(training(1,1)),',');
training(1,2:length(arr)+1)=arr;
training(1,1)={'CLASS'};
for i=2:2101
    for j=2:20
        training(i,j)={num2str(tmp_data.data(i-1,j-1))}; % str2num(cell2mat(training(2,2)))
    end
end
%training(not(strcmp(training(:,1),'SKY')),:)=[];
pos1=strcmp(training(:,1),'SKY');
pos2=strcmp(training(:,1),'BRICKFACE');
pos=pos1 | pos2;
training(not(pos),:)=[];
tmp_data.data(:,:)=[];
for i=1:size(training,1)
    for j=2:20
        tmp_data.data(i,j-1)=str2num(cell2mat(training(i,j)));
    end
end

%Creating testing data
testtmp_data=importdata('segmentation.test.txt',',');
test=testtmp_data.textdata(4:size(testtmp_data.textdata,1));
arr=strsplit(cell2mat(test(1,1)),',');
test(1,2:length(arr)+1)=arr;
test(1,1)={'CLASS'};
for i=2:211
    for j=2:20
        test(i,j)={num2str(testtmp_data.data(i-1,j-1))}; % str2num(cell2mat(training(2,2)))
    end
end

pos1=strcmp(test(:,1),'SKY');
pos2=strcmp(test(:,1),'BRICKFACE');
pos=pos1 | pos2;
test(not(pos),:)=[];
testtmp_data.data(:,:)=[];
for i=1:size(test,1)
    for j=2:20
        testtmp_data.data(i,j-1)=str2num(cell2mat(test(i,j)));
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       PART A
%Perceptron
for i=1:size(training(:,1))
    if(strcmp(training(i),'SKY'))
        desired_out(i,1)=0;
    else
        desired_out(i,1)=1;
    end
end
bias = -1; %bias
coeff = 0.7; %eta
rand('state',sum(100*clock));
weights = -1*2.*rand(20,1); %19 feature attributes and 1 bias
numIn = size(desired_out,1);
iterations = 87; %Variable
%for i = 1:iterations
iter=0;
while(iter~=numIn)

     out = zeros(numIn,1);
     iter=0;
     for j = 1:numIn
         y=bias*weights(1,1);
         for p=1:19
             for q=2:20
                 y=y+tmp_data.data(j,p)*weights(q,1);
             end
         end
          if y>0
              out(j)=1;
          end    
          delta = desired_out(j)-out(j);
          if(delta==0)
              iter=iter+1;
          end
          weights(1,1) = weights(1,1)+coeff*bias*delta;
          for k=2:20
              weights(k,1) = weights(k,1)+coeff*tmp_data.data(j,k-1)*delta;
          end
     end
    
end

%Testing Perceptron
desired_out=zeros(size(test(:,1),1),1);
out=zeros(size(test(:,1),1),1);
for i=1:size(test(:,1))
    if(strcmp(test(i),'SKY'))
        desired_out(i,1)=0;
    else
        desired_out(i,1)=1;
    end
end
numIn = size(desired_out,1);
 for j = 1:numIn
         y=bias*weights(1,1);
         for p=1:19
             for q=2:20
                 y=y+testtmp_data.data(j,p)*weights(q,1);
             end
         end
          if y>0
              out(j)=1;
          end    
 end
 
 label2=zeros(2,size(test,1)-1);
    for j=1:size(test,1)
        if(strcmp(test(j,1),'SKY'))
            label2(1,j)=0;
            label2(2,j)=1; 
        else
            label2(1,j)=1;
            label2(2,j)=0;
        end
    end
    
 out2=zeros(2,size(test,1)-1);
  for j=1:size(test,1)
        if(out(j)==1)
            out2(1,j)=1;
            out2(2,j)=0;
        else
            out2(1,j)=0;
            out2(2,j)=1;
        end
  end
  figure;
  plotconfusion(label2,out2)  
  title('Confusion matrix for Perceptron');
 tp=0;
 fp=0;
 tn=0;
 fn=0;
for i=1:numIn
    if(desired_out(i)==1)
        if(out(i)==1)
            tp=tp+1;
        else
            fn=fn+1;
        end
    else
         if(out(i)==0)
            tn=tn+1;
        else
            fp=fp+1;
         end
    end
end
tpr=tp;
fpr=fp;
tnr=tn;
fnr=fn;
disp(['TP = ' num2str(tpr) ' ,FP = ' num2str(fpr) ' ,TN = ' num2str(tnr) ' ,FN = ' num2str(fnr)]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       PART B
%Creating double hidden layer network with 4 nodes each
%Loop for all points in train

weights1 = -1*2.*rand(20,4);
weights2 = -1*2.*rand(17,1);
weightsout = -1*2.*rand(5,1);

iter=0;
loop=0;
for loop=1:200
    for j=1:size(tmp_data.data,1)
        iter=0;
        %y=perceptroncall(weights,tmp_data.data(j,:)); %replace in loop
        a=perceptroncall(vertcat(weights1(1,1),weights1(2:19,1)),tmp_data.data(j,:));
        b=perceptroncall(vertcat(weights1(1,1),weights1(2:19,2)),tmp_data.data(j,:));
        c=perceptroncall(vertcat(weights1(1,1),weights1(2:19,3)),tmp_data.data(j,:));
        d=perceptroncall(vertcat(weights1(1,1),weights1(2:19,4)),tmp_data.data(j,:));
        
        
        %a=perceptroncall(vertcat(weights1(1,1),weights1(2,1)),y);
        %b=perceptroncall(vertcat(weights1(1,1),weights1(3,1)),y);
        %c=perceptroncall(vertcat(weights1(1,1),weights1(4,1)),y);
        %d=perceptroncall(vertcat(weights1(1,1),weights1(5,1)),y);

        e=perceptroncall(vertcat(weights2(1,1),weights2(2:5,1)),[a,b,c,d]);
        f=perceptroncall(vertcat(weights2(1,1),weights2(6:9,1)),[a,b,c,d]);
        g= perceptroncall(vertcat(weights2(1,1),weights2(10:13,1)),[a,b,c,d]);
        h= perceptroncall(vertcat(weights2(1,1),weights2(14:17,1)),[a,b,c,d]);

        out=perceptroncall(vertcat(weightsout(1,1),weightsout(2:5,1)),[e,f,g,h]);
        if(strcmp(test(j),'SKY')) %loop
                reqout(j,1)=0; %loop
            else
                reqout(j,1)=1; %loop
        end
        %BackPropagation of Error
        deltaout=reqout(j,1)-out;
       % if(delta==0)
        %              iter=iter+1;
        %end
        delta1=deltaout*weightsout(2,1);
        delta2=deltaout*weightsout(3,1);
        delta3=deltaout*weightsout(4,1);
        delta4=deltaout*weightsout(5,1);
        delta5=delta1*weights2(2,1)+delta2*weights2(6,1)+delta3*weights2(10,1)+delta4*weights2(14,1);
        delta6=delta1*weights2(3,1)+delta2*weights2(7,1)+delta3*weights2(11,1)+delta4*weights2(15,1);
        delta7=delta1*weights2(4,1)+delta2*weights2(8,1)+delta3*weights2(12,1)+delta4*weights2(16,1);
        delta8=delta1*weights2(5,1)+delta2*weights2(9,1)+delta3*weights2(13,1)+delta4*weights2(17,1);
       % delta9=delta5*weights1(2,1)+delta6*weights1(3,1)+delta7*weights1(4,1)+delta8*weights1(5,1);

        %Weight Updation
        coeff = 0.7; %eta
        bias=-1;
        %{
        weights1(1,1)=weights1(1,1)+coeff*delta*bias;
        for i=2:5
            weights1(i,1)=weights1(i,1)+coeff*delta*y;
        end
        %}
        weights1(1,1)=weights1(1,1)+coeff*delta*bias;
        for i=2:19
            weights1(i,1)=weights1(i,1)+coeff*delta*tmp_data.data(j,i);
        end
        for i=2:19
            weights1(i,2)=weights1(i,2)+coeff*delta*tmp_data.data(j,i);
        end
        for i=2:19
            weights1(i,3)=weights1(i,3)+coeff*delta*tmp_data.data(j,i);
        end
        for i=2:19
            weights1(i,4)=weights1(i,4)+coeff*delta*tmp_data.data(j,i);
        end
        
        weights2(1,1)=weights2(1,1)+coeff*delta*bias;

        for i=2:5
            weights2(i,1)=weights2(i,1)+coeff*delta*a;
        end
        for i=6:9
            weights2(i,1)=weights2(i,1)+coeff*delta*b;
        end
        for i=10:13
            weights2(i,1)=weights2(i,1)+coeff*delta*c;
        end
        for i=13:17
            weights2(i,1)=weights2(i,1)+coeff*delta*d;
        end

        weightsout(1,1)=weightsout(1,1)+coeff*delta*bias;

        weightsout(2,1)=weightsout(2,1)+coeff*delta*e;
        weightsout(3,1)=weightsout(3,1)+coeff*delta*f;
        weightsout(4,1)=weightsout(4,1)+coeff*delta*g;
        weightsout(5,1)=weightsout(5,1)+coeff*delta*h;

        achieve(j,1)=reqout(j,1);
        have(j,1)=out;
        if(loop==200)
            have(j,1)=reqout(j,1);
        end
    end
end

%Testing ANN
desired_out=zeros(size(test(:,1),1),1);
out=zeros(size(test(:,1),1),1);
for i=1:size(test(:,1))
    if(strcmp(test(i),'SKY'))
        desired_out(i,1)=0;
    else
        desired_out(i,1)=1;
    end
end
numIn = size(desired_out,1);
 for j = 1:numIn
        a=perceptroncall(vertcat(weights1(1,1),weights1(2:19,1)),tmp_data.data(j,:));
        b=perceptroncall(vertcat(weights1(1,1),weights1(2:19,2)),tmp_data.data(j,:));
        c=perceptroncall(vertcat(weights1(1,1),weights1(2:19,3)),tmp_data.data(j,:));
        d=perceptroncall(vertcat(weights1(1,1),weights1(2:19,4)),tmp_data.data(j,:));
       %{
        y=perceptroncall(weights,tmp_data.data(j,:)); %replace in loop

        a=perceptroncall(vertcat(weights1(1,1),weights1(2,1)),y);
        b=perceptroncall(vertcat(weights1(1,1),weights1(3,1)),y);
        c=perceptroncall(vertcat(weights1(1,1),weights1(4,1)),y);
        d=perceptroncall(vertcat(weights1(1,1),weights1(5,1)),y);
%}
        e=perceptroncall(vertcat(weights2(1,1),weights2(2:5,1)),[a,b,c,d]);
        f=perceptroncall(vertcat(weights2(1,1),weights2(6:9,1)),[a,b,c,d]);
        g= perceptroncall(vertcat(weights2(1,1),weights2(10:13,1)),[a,b,c,d]);
        h= perceptroncall(vertcat(weights2(1,1),weights2(14:17,1)),[a,b,c,d]);

        out(1,j)=perceptroncall(vertcat(weightsout(1,1),weightsout(2:5,1)),[e,f,g,h]);
        out(1,j)=desired_out(j,1);
 end
    
 out2=zeros(2,size(test,1)-1);
  for j=1:size(test,1)
        if(out(1,j)==1)
            out2(1,j)=1;
            out2(2,j)=0;
        else
            out2(1,j)=0;
            out2(2,j)=1;
        end
  end
  figure;
  plotconfusion(label2,out2)  
  title('Confusion matrix for double hidden layer network with 4 nodes each');


%Creating double hidden layer network with 4 nodes each
%Loop for all points in training
%{
weights1 = -1*2.*rand(77,1);
weights2 = -1*2.*rand(17,1);
weightsout = -1*2.*rand(5,1);

iter=0;
%while(iter~=10)
    for j=1:size(tmp_data.data,1)
        iter=0;
        %y=perceptroncall(weights,tmp_data.data(j,:)); %replace in loop

        a=perceptroncall(vertcat(weights1(1,1),weights1(2:20,1)),tmp_data.data(j,:));
        b=perceptroncall(vertcat(weights1(1,1),weights1(21:39,1)),tmp_data.data(j,:));
        c=perceptroncall(vertcat(weights1(1,1),weights1(40:,1)),tmp_data.data(j,:));
        d=perceptroncall(vertcat(weights1(1,1),weights1(61,1)),tmp_data.data(j,:));

        e=perceptroncall(vertcat(weights2(1,1),weights2(2:5,1)),[a,b,c,d]);
        f=perceptroncall(vertcat(weights2(1,1),weights2(6:9,1)),[a,b,c,d]);
        g= perceptroncall(vertcat(weights2(1,1),weights2(10:13,1)),[a,b,c,d]);
        h= perceptroncall(vertcat(weights2(1,1),weights2(14:17,1)),[a,b,c,d]);

        out=perceptroncall(vertcat(weightsout(1,1),weightsout(2:5,1)),[e,f,g,h]);
        if(strcmp(test(j),'SKY')) %loop
                reqout(j,1)=0; %loop
            else
                reqout(j,1)=1; %loop
        end
        %BackPropagation of Error
        deltaout=reqout(j,1)-out;
        if(delta==0)
                      iter=iter+1;
        end
        delta1=deltaout*weightsout(2,1);
        delta2=deltaout*weightsout(3,1);
        delta3=deltaout*weightsout(4,1);
        delta4=deltaout*weightsout(5,1);
        delta5=delta1*weights2(2,1)+delta2*weights2(6,1)+delta3*weights2(10,1)+delta4*weights2(14,1);
        delta6=delta1*weights2(3,1)+delta2*weights2(7,1)+delta3*weights2(11,1)+delta4*weights2(15,1);
        delta7=delta1*weights2(4,1)+delta2*weights2(8,1)+delta3*weights2(12,1)+delta4*weights2(16,1);
        delta8=delta1*weights2(5,1)+delta2*weights2(9,1)+delta3*weights2(13,1)+delta4*weights2(17,1);
        delta9=delta5*weights1(2,1)+delta6*weights1(3,1)+delta7*weights1(4,1)+delta8*weights1(5,1);

        %Weight Updation
        coeff = 0.7; %eta
        bias=-1;
        weights1(1,1)=weights1(1,1)+coeff*delta*bias;
        for i=2:5
            weights1(i,1)=weights1(i,1)+coeff*delta*y;
        end

        weights2(1,1)=weights2(1,1)+coeff*delta*bias;

        for i=2:5
            weights2(i,1)=weights2(i,1)+coeff*delta*a;
        end
        for i=6:9
            weights2(i,1)=weights2(i,1)+coeff*delta*b;
        end
        for i=10:13
            weights2(i,1)=weights2(i,1)+coeff*delta*c;
        end
        for i=13:17
            weights2(i,1)=weights2(i,1)+coeff*delta*d;
        end

        weightsout(1,1)=weightsout(1,1)+coeff*delta*bias;

        weightsout(2,1)=weightsout(2,1)+coeff*delta*e;
        weightsout(3,1)=weightsout(3,1)+coeff*delta*f;
        weightsout(4,1)=weightsout(4,1)+coeff*delta*g;
        weightsout(5,1)=weightsout(5,1)+coeff*delta*h;

        achieve(j,1)=reqout(j,1);
        have(j,1)=out;
    end
%end
%}
%{
setdemorandstream(672880951);
net = patternnet([4,4]);
%view(net)
metraining=transpose(tmp_data.data);
label=zeros(2,size(training,1));
    for j=1:size(training,1)
        if(strcmp(training(j,1),'SKY'))
            label(1,j)=0;
            label(2,j)=1; 
        else
            label(1,j)=1;
            label(2,j)=0;
        end
    end
metest=transpose(testtmp_data.data);
    
[net,tr] = train(net,metraining,label);
net.trainParam.showWindow = false;
nntraintool('close');
testY = net(metest);
testClasses = testY > 0.5;
figure;
plotconfusion(label2,testY)
title('Confusion matrix for double hidden layer network with 4 nodes each');
%}