function [FV] = FindFiedlerVector(A)
% finds the community structure over a graph based on the eigenvector
% corresponding to the second smallest eigenvalue of the graph laplacian. 
Deg = sum(A,1);
L = diag(Deg)-A;
[V,D] = eig(full(L));
[~,idx] = sort(diag(D));
V=V(:,idx);
V = V(:,2);
FV = zeros(size(A,1),1);
FV(V>=0)=1;
FV(V<0)=2;
end