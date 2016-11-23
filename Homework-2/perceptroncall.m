function [ out ] = perceptroncall( weights,data )

bias = -1; %bias
y=bias*weights(1,1);
      for p=1:(size(weights,1)-1)
          for q=2:size(weights,1)
                %disp([num2str(data(1,p)) ' ' num2str(weights(q,1))]);
                y=y+data(1,p)*weights(q,1);
          end
      end
      %disp([num2str(y)]);
out=y;
end
