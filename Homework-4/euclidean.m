function [ dis ] = euclidean( x,y )
sum=0;
for i=1:length(x)
    sum=sum+(y(i)-x(i))^2;
end
dis=sqrt(sum);
end

