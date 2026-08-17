function [C] = GetConductance(S,A)
% Calculates conductance for a partition S over adjacency matrix A. 
CS = unique(S);
C = zeros([1,size(CS,1)]);
for i = 1:size(CS,1)
C(i)=sum(sum(A(S==CS(i),S~=CS(i)))) / min(sum(S==CS(i)),ceil(size(A,1)/2));
end
C = min(C);
end