function sim = computeSim(ref, val)
 A = real(ifft2(ref));
 B = real(ifft2(val));
 %C = power(sum(A,3) - sum(B,3),2);
 C = power(sum(A,3) - sum(B,3),2);
 d = sqrt(sum(C(:)));        
 sim = exp(-0.05*d);