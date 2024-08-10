function submat = kern(radius, order, srcinfo, targinfo, varargin)
    %QBX.KERN   Evaluate the QBX kernel Laplace 2D 

    % WHAT IS varagin?
    
    % The Greens function of the Dirac operator (after Fourier transform in z)
    % is given by the sum of three components:
    
    % 1. gamma_1 .* (\nabla_{x} G_{zk}(x,y) \dot [1;0])
    % 2. gamma_2 .* (\nabla_{x} G_{zk}(x,y) \dot [1;0])
    % 3. [m.*gamma0 + w.*gamma5 + E.*I_4 + i*\xi.*gamma3 ] .* G_{zk}(x,y)
    
    % The sum of the above kernel should be scaled by (2*pi)^(-1/2) due to the
    % Fourier transform
    
    
    % Input:
    %   r - expansion radius
    %   srcinfo - description of sources in ptinfo struct format, i.e.
    %                ptinfo.r - positions (2,:) array
    %                ptinfo.d - first derivative in underlying
    %                     parameterization (2,:)
    %                ptinfo.d2 - second derivative in underlying
    %                     parameterization (2,:)
    %   targinfo - description of targets in ptinfo struct format,
    %                if info not relevant (d/d2) it doesn't need to
    %                be provided. sprime requires tangent info in
    %                targinfo.d
    %
    % Output:
    %   submat - the evaluation of the kernel for the
    %            provided sources and targets. the number of
    %            rows equals the number of targets and the
    %            number of columns equals the number of sources  
    %
    % see also QBX.GREEN
    
    
    
    submat = dirac2d.green(radius, order, srcinfo, targinfo, varargin);
    
    end
    
    
    
    
    
    