function ret = K_opt_min(det, to_min, mean_cons)
    function ret = sympy_to_numpy(variables)
        ret = @(expr) matlabFunction(expr, variables);
    end

    function ret = array_stretcher_helper(arr, from_i)
        res = [];
        c = 1;
        for i = arr
            res = [res i];
            if c >= from_i
                res = [res 0];
            end
            c = c + 1;
        end
        ret = res;
    end

    array_stretcher = @(f, from_i) ( @(arg) array_stretcher_helper(f(arg), from_i));

    function ret = lambda_out_array_helper(expr) 
        ret = @(arg) cell2mat(map( @(e) e(arg), expr));
    end

    lambda_out_array = @(arg, from_i) array_stretcher(lambda_out_array_helper(arg), from_i);

    function r = array_to_args(f, arg)
        args = num2cell(arg);
        [r, ~] = f(args{:});
    end

    function ret = complex_array_wrapper(arg, from_i)
        c = 1;
        for i = 1 : from_i-1
            b(c) = arg(i);
            c = c + 1;
        end
        for i = from_i : 2 : length(arg)
            b(c) = arg(i) + arg(i+1) * 1j;
            c = c + 1;
        end
        ret = b;
    end

    function_wrapper = @(f, from_i) ( @(arg) array_to_args(f, complex_array_wrapper(arg, from_i)));

    complex_wrapper = @(f, from_i) ( @(arg) f(complex_array_wrapper(arg, from_i)));


    function ret = map(f, list)
        ret = cell(1, length(list));
        for i = 1:length(list)
            ret{i} = f(list{i});
        end
    end

    function ret = diff_sympy_func(variables)
        function r = f(expr)
            r = cell(1, length(variables));
            for i = 1:length(variables)
                r{i} = diff(expr, variables(i));
            end
        end
        ret = @f;
    end
        
    
    global eta alpha1 alpha2 beta1 beta2 theta psi1 psi2 psi3 psi4 r omega
    vars = [theta, psi1, psi2, psi3, psi4];
    sympy_2_numpy = sympy_to_numpy(vars);
    deriv_sympy_func = diff_sympy_func(vars);
    n_det = sympy_2_numpy(-real(det));
    deriv_sympy = deriv_sympy_func(-real(det));
    deriv_numpy = lambda_out_array(map( @(x) function_wrapper(sympy_2_numpy(x), 2), deriv_sympy), 2);
    function r = func_to_min(args)
        f = function_wrapper(sympy_2_numpy(to_min), 2);
        r = real(f(args));
        res = lambda_out_array(map( @(x) function_wrapper(sympy_2_numpy(x), 2), deriv_sympy_func(to_min)), 2);
    end
    sqrt_8 = 1 / sqrt(8);
    x0 =  [pi/3.0, sqrt_8, sqrt_8, sqrt_8, sqrt_8, sqrt_8, sqrt_8, sqrt_8, sqrt_8];
    psi_cons = complex_wrapper( @(p) sum(conj(p(2:end)) .* p(2:end)) - 1, 2);
    psi_cons_deriv = array_stretcher(complex_wrapper( @(x) [0, 2*abs(x(2)), 2*abs(x(3)), 2*abs(x(4)), 2*abs(x(5))], 2), 2);
    function [c,ceq, gc, gceq] = mycon(x)
        f = function_wrapper(n_det, 2);
        c =  f(x) % Compute nonlinear inequalities at x.
        ceq = psi_cons(x) % Compute nonlinear equalities at x.
        x
        gc = deriv_numpy(x)'
        gceq = psi_cons_deriv(x)'
    end
    options = optimset('Algorithm', 'interior-point', 'Display','iter','GradObj', 'off', 'GradConstr', 'on');
    res = fmincon(@func_to_min, x0, [], [], [], [], [], [], @mycon, options)
    %jac=deriv_func_to_min,\
    %constraints=cons, method='SLSQP', options={'disp': True})
    
%     vars = [theta];
%     sympy_2_numpy = sympy_to_numpy(vars);
%     deriv_sympy_func = diff_sympy_func(vars);
%     n_det = sympy_2_numpy(real(det));
%     det_func = function_wrapper(n_det, 2);
%     deriv_sympy = deriv_sympy_func(real(det));
%     deriv_numpy = lambda_out_array(map( @(x) function_wrapper(sympy_2_numpy(x), 2), deriv_sympy), 2);
%     vars = [r, omega];
%     sympy_2_numpy = sympy_to_numpy(vars);
%     deriv_sympy_func = diff_sympy_func(vars);
%     res = fminsearch(det_func, 0, optimset('TolFun', 1e-12));
%     to_subs = theta;
%     to_subs_val = res;
%     subs_to_min = subs(to_min, to_subs, to_subs_val);
%     to_min_func = function_wrapper(sympy_2_numpy(subs_to_min), 3);
%     res = fminsearch(to_min_func, [1, 0], optimset('Display', 'iter'))

%     function [r, gr] =  func_to_min(args)
%         f = function_wrapper(sympy_2_numpy(to_min), 4);
%         r = real(f(args));
%         res = lambda_out_array(map( @(x) function_wrapper(sympy_2_numpy(x), 4), deriv_sympy_func(f)), 4);
%         gf = res(args);
%         gr = zeros(1, length(gf));
%         for i = 1:length(gf)
%             gr(i) = real(gf{i});
%         end
%     end
%     x0 = [pi/3.0, 0.05, 0.0001];
%     function [c,ceq, gc, gceq] = mycon(x)
%         c =  det_func(x); % Compute nonlinear inequalities at x.
%         ceq = 0; % Compute nonlinear equalities at x.
%         det_grad = deriv_numpy(x);
%         for i = 1 : length(det_grad)
%             gc(i, 1) = det_grad{i};
%         end
%         gceq = [0; 0; 0];
%     end
%     lb = [0, 0, 0];
%     ub = [2*pi, 1, 2*pi];
%     options = optimset('Algorithm', 'interior-point', 'Display','iter','GradObj', 'on', 'GradConstr', 'on');
%     res = fmincon(@func_to_min, x0, [], [], [], [], lb, ub, @mycon, options)
    ret = res ;
end

