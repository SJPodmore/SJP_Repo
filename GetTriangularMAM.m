function [M] = GetTriangularMAM(A,motif)
% Calculates the MAM of a given motif over the graph A by walks. 
%% Initialization
B = A.*A'; %Bidirection matrix
Bl = tril(B); %Lower triangular bidirectional matrix 
U = A-B; %Unidirectional edge matrix
G = double(full((A+A'>0))); %Symmetrised matrix
Ac = (1-G-eye(size(A))); %Graph Compliment
Acl = tril(Ac); %Lower triangular graph compliment

switch motif 
    case "T78"
        M = (Bl*B).*Ac + (Bl'*B).*Ac + (B*Acl).*B + (B*Acl').*B + (Acl*B).*B + (Acl'*B).*B;
    case {"T78A23","T78A32"}
        M = (Bl*B).*Ac + (Bl'*B).*Ac;
    case {"T78A12","T78A21","T78A13","T78A31"}
        %Isomorphic to T78A13
        M = (B*Acl).*B + (B*Acl').*B + (Acl*B).*B + (Acl'*B).*B;
    case "T74"
        M = (Bl*U').*Ac + (Bl'*U').*Ac + (U'*Acl).*B + (U'*Acl').*B + (Acl*B).*U + (Acl'*B).*U;
        M = M+M';
    case {"T74A23","T74A32"}
        M = (Bl*U').*Ac + (Bl'*U').*Ac;
        M = M+M';
    case {"T74A12","T74A21"}
        M = (U'*Acl).*B + (U'*Acl').*B;
        M = M+M';
    case {"T74A13","T74A31"}
        M = (Acl*B).*U + (Acl'*B).*U;
        M = M+M';
    case "T14"
        M = (Bl*U).*Ac + (Bl'*U).*Ac + (U*Acl).*B + (U*Acl').*B + (Acl*B).*U' + (Acl'*B).*U';
        M = M+M';
    case {"T14A12","T14A21"}
        M = (U*Acl).*B + (U*Acl').*B;
        M = M+M';
    case {"T14A13","T14A31"}
        M = (Acl*B).*U' + (Acl'*B).*U';
        M = M+M';
    case {"T14A23","T14A32"}
        M = (Bl*U).*Ac + (Bl'*U).*Ac;
        M = M+M';
    case "T36"
        M = (U*U').*Ac + (U'*Acl).*U' + (U'*Acl').*U' + (Acl*U).*U + (Acl'*U).*U;
    case {"T36A23","T36A32"}
        M = (U*U').*Ac;
    case {"T36A12","T36A21","T36A13","T36A31"}
        %Isomorphic to T36A13
        M = (U'*Acl).*U' + (U'*Acl').*U' + (Acl*U).*U + (Acl'*U).*U;
    case "T12"
        M = (U*U).*Ac + (U*Acl).*U' + (U*Acl').*U' + (Acl*U).*U' + (Acl'*U).*U';
        M = M+M';
    case {"T12A23","T12A32"}
        M = (U*U).*Ac;
        M = M+M';
    case {"T12A12","T12A21"}
        M = (Acl*U).*U' + (Acl'*U).*U';
        M = M+M';
    case {"T12A13","T12A31"}
        M = (U*Acl).*U' + (U*Acl').*U';
        M = M+M';
    case "T6"
        M = (Acl*U').*U' + (Acl'*U').*U' + (U*Acl).*U + (U*Acl').*U + (U'*U).*Ac;
    case {"T6A12","T6A21","T6A13","T6A31"}
        %Isomorphic to T6A13
        M = (Acl*U').*U' + (Acl'*U').*U'+ (U*Acl).*U + (U*Acl').*U;
    case {"T6A23","T6A32"}
        M = (U'*U).*Ac;
    case "T46"
        M = (U'*Bl).*U' + (U'*Bl').*U'+(Bl*U).*U + (Bl'*U).*U + (U*U').*B;
    case {"T46A12","T46A13","T46A21","T46A31"}
        % Isomorphic to T46A13
        M = (U'*Bl).*U' + (U'*Bl').*U'+(Bl*U).*U + (Bl'*U).*U;
    case {"T46A23","T46A32"}
        M =  (U*U').*B;
    case "T108"
        M = (U*Bl).*U + (U*Bl').*U + (Bl*U').*U' + (Bl'*U').*U' + (U'*U).*B;
    case {"T108A23","T108A32"}
        M = (U'*U).*B;
    case {"T108A12","T108A21","T108A13","T108A31"}
        % Isomorphic to T108A13
        M = (U*Bl).*U + (U*Bl').*U + (Bl*U').*U' + (Bl'*U').*U';
    case "T38"
        M = (U*U).*U + (U*U').*U + (U'*U).*U;
        M = M+M';
    case {"T38A12","T38A21"}
        M = (U*U').*U;
        M = M+M';
    case {"T38A23","T38A32"}
        M = (U'*U).*U;
        M = M+M';
    case {"T38A13","T38A31"}
        M = (U*U).*U;
        M = M+M';
    case "T110"
        M = (Bl*B).*U + (Bl'*B).*U + (Bl*U).*B + (Bl'*U).*B + (U*Bl).*B+(U*Bl').*B;
        M = M+M';
    case {"T110A23","T110A32"}
        M= (Bl*B).*U + (Bl'*B).*U;
        M = M+M';
    case {"T110A12","T110A21"}
        M = (U*Bl).*B+(U*Bl').*B;
        M = M+M';
    case {"T110A13","T110A31"}
        M = (Bl*U).*B + (Bl'*U).*B ;
        M = M+M';
    case "T102"
        M = (Bl*U).*U' + (Bl'*U).*U' + (U*Bl).*U' + (U*Bl').*U' + (U*U).*B;
        M = M+M';
    case {"T102A13","T102A31"}
        M = (U*U).*B;
        M = M+M';
    case {"T102A23","T102A32"}
        M = (Bl*U).*U' + (Bl'*U).*U';
        M = M+M';
    case {"T102A12","T102A21"}
        M = (U*Bl).*U' + (U*Bl').*U';
        M = M+M';
    case "T238"
        M = (B*B).*B;
    case "T98"
        M = (U*U).*U';
        M = M+M';
    otherwise
        M=0;
        print("No motif found")
end
end