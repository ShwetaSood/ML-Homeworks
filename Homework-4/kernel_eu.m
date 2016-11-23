function [ dis ] = kernel_eu( x,y,i,K_matrix,class)

sum1=K_matrix(i,i);
sum2=0; sum3=0;
for j=1:length(class)
    if(class(j)~=0)
        sum2=sum2+K_matrix(i,class(j));
    end
end

for k=1:length(class)
   for j=1:length(class)
       if(class(j)~=0)
        sum3=sum3+K_matrix(class(k),class(j));
       end
   end
end

dis=sum1-((2*sum2)/length(class))+(sum3/(length(class)^2));

end

