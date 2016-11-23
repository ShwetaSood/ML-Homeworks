clear all;
clc;
addpath('C:\Users\ShwetaS\Desktop\ML\HW\ShwetaSood_2012164_Ass2\libsvm-3.20\matlab\');
class1 = [5 12; 5 -12; -5 12; -5 -12; 0 13; 0 -13; 13 0; -13 0]; 
class2 = [0 5; 5 0; -5 0; 0 -5; 4 3; -4 3; 4 -3; -4 -3];
figure;
scatter(class1(:,1),class1(:,2),'filled','MarkerFaceColor','blue');
hold on;
scatter(class2(:,1),class2(:,2),'filled','MarkerFaceColor','red');
legend('Class 1','Class 2');
xlabel('X coordinate');
ylabel('Y coordinate');
title('Distribution of data for classes 1 and 2');
class=vertcat(class1,class2);
label=ones(length(class1),1);
label=vertcat(label,ones(length(class1),1)+1);

%{
%OPTION 1
bestcv = 0;
accuracy=zeros(1,5);
cval=zeros(1,5);
k=1;
for c_val = -1:3
    for g_val=-4:1
        cv = svmtrain(label, class, ['-t 2 -v 5 -c ', num2str(2^c_val), ' -g ', num2str(2^g_val)]);
        if (cv >= bestcv),
          bestcv = cv; bestc = 2^c_val; bestg = 2^g_val;
        end
        fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', c_val, g_val, cv, bestc, bestg, bestcv);
        if(bestcv==100)
            break;
        end
    end
    if(bestcv==100)
        break;
    end
end
%}
%OPTION 2
trainclass=[0 13;0 -13;13 0;-13 0;0 5;0 -5;-5 0;5 0];
testclass=[5 12; 4 3;5 -12; -4 3;-5 12; 4 -3;-5 -12;-4 -3;];
trainlabels=[1;1;1;1;2;2;2;2];
testlabels=[1;2;1;2;1;2;1;2];
bestcv = 0;

for c_val = -1:6
    for g_val=-4:1
        model = svmtrain(trainlabels,trainclass, ['-t 2 -c ', num2str(2^c_val) , ' -g ', num2str(2^g_val)]);
        [predicted_label, acc, prob_estimates]=svmpredict(testlabels,testclass,model);
        if (acc(1) >= bestcv)
          bestcv = acc(1); bestc = 2^c_val; bestg=2^g_val;
        end
        fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', c_val, g_val, acc(1), bestc, bestg, bestcv);
        if(bestcv==100)
            break;
        end
    end
    if(bestcv==100)
        disp(['100% accuracy achieved with Kernel RBF with above parameters. Support vectors are <x,y>:']);
        break;
    end
end
for i=1:model.totalSV
    index=model.sv_indices(i);
    disp([num2str(trainclass(index,1)),' ' ,num2str(trainclass(index,2))]);
end
%{
for log2c = -1:6
        cmd = ['-t 0 ', '-c ', num2str(2^log2c) ];
        model = svmtrain(trainlabels,trainclass, cmd);
        [predicted_label, acc, prob_estimates]=svmpredict(testlabels,testclass,model);
        if (acc(1) >= bestcv)
          bestcv = acc(1); bestc = 2^log2c;
        end
        fprintf('%g  %g (best c=%g, rate=%g)\n', log2c,  acc(1), bestc,  bestcv);
        if(bestcv==100)
            break;
        end
end
%}