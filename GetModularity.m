function [Q] = GetModularity(S,A)
%Calculates the modularity for the partition S over adjacency matrix A.
%Only works for undirected graphs. 
m = sum(sum(A));
Q = 0;
CS = unique(S);
for j=1:length(CS)
    Cj = find(S==CS(j));
    Int = sum(sum(A(Cj,Cj)));
    Ext = sum(sum(A(Cj,:)));
    if Et>0
        Q = Q + Int/m-(Ext/m)^2;
    end
end