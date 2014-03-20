function ret = basics()
    ones(10)*ones(10);
    function ret = get_sigma(theta)
        e = sym(exp(1), 'd')^(2*1j*theta);
        sigma = [0 e 0 0; conj(e) 0 0 0; 0 0 0 e; 0 0 conj(e) 0];
        ret = sigma;
    end

    function ret = get_tau(theta)
        e = sym(exp(1), 'd')^(2*1j*theta);
        tau = [0 0 e 0; 0 0 0 e; conj(e) 0 0 0; 0 conj(e) 0 0];
        ret = tau;
    end

    function ret = get_n_o(eta, theta, is_first)
        if is_first
            ret = eta / 2.0 * (eye(4) + get_sigma(theta));
        else
            ret = eta / 2.0 * (eye(4) + get_tau(theta));
        end
    end

    function ret = get_n_e(eta, theta, is_first)
        if is_first
            ret = eta / 2.0 * (eye(4) - get_sigma(theta));
        else
            ret = eta / 2.0 * (eye(4) - get_tau(theta));
        end
    end

    function ret = get_n_u(eta, theta, is_first)
        if is_first
            ret = (1 - eta);
        else
            ret = (1 - eta);
        end
    end

    function ret = get_B(eta, theta)
        A = eta / 2.0 * (sym(exp(1), 'd')^(-2 * 1j * theta) - 1);
        B = exp(1)^(-2 * I * theta) - 1;
        tmp = 1 - eta;
        M = eta / 2.0 * [(2 - eta) tmp tmp (conj(A)*conj(B) - eta);
                         tmp (2 - eta) (A*conj(B) - eta) tmp;
                         tmp (conj(A)*B - eta) (2 - eta) tmp;
                         (A*B - eta) tmp tmp (2 - eta)];
        ret = M;
    end

    %function ret = eigenvalues_analyse()
    %    eigenvalues = [k for (k, v) in n.eigenvals().iteritems()]
    %    ret = eigenvalues
    
    global eta alpha1 alpha2 beta1 beta2 theta psi1 psi2 psi3 psi4 r omega
    eta = sym('eta', 'positive');
    theta = sym('theta', 'positive');
    alpha1 = sym('alpha_1');
    alpha2 = sym('alpha_2');
    beta1 = sym('beta_1');
    beta2 = sym('beta_2');
    eta1 = eta;
    eta2 = eta ;
    n_oo = @(alpha, beta) get_n_o(eta1, alpha, true) * get_n_o(eta2, beta, false);
    n_oe = @(alpha, beta) get_n_o(eta1, alpha, true) * get_n_e(eta2, beta, false);
    n_ou = @(alpha, beta) get_n_o(eta1, alpha, true) * get_n_u(eta2, beta, false);
    n_eo = @(alpha, beta) get_n_e(eta1, alpha, true) * get_n_o(eta2, beta, false);
    n_uo = @(alpha, beta) get_n_u(eta1, alpha, true) * get_n_o(eta2, beta, false);
    n = n_oe(0, theta) + n_ou(0, theta) + n_eo(theta, 0) + n_uo(theta, 0) + n_oo(theta, theta) - n_oo(0, 0);
    psi1 = sym('psi_1');
    psi2 = sym('psi_2');
    psi3 = sym('psi_3');
    psi4 = sym('psi_4');
    r = sym('r', 'positive');
    omega = sym('omega', 'positive');
    %psi = sym(1 / 2.0, 'd') / sqrt(1 + r^2) * [(1+r)*exp(1)^(-1j*omega); -(1-r); -(1-r); (1+r)*exp(1)^(1j*omega)];
    psi = [psi1; psi2; psi3; psi4];
    quant_mean = psi' * n * psi;
    quant_std = psi' * n * n * psi - quant_mean*quant_mean;
    to_min = quant_mean;% / sqrt(quant_std);
    function ret = eta_K_opt(det, to_min, mean_cons)
        etas =  0.01* (66:1:100);
        res = [];
        for e1 = etas
            e1
            to_subs = [eta1 eta2];
            to_subs_val = [e1, e1];
            subs_det = subs(det, to_subs, to_subs_val);
            det_func = matlabFunction(subs_det);
            %ezplot(det_func, [0, pi])
            subs_to_min = subs(to_min, to_subs, to_subs_val);
            subs_mean_cons = subs(mean_cons, to_subs, to_subs_val);
            res = [res K_opt_min(subs_det, subs_to_min, subs_mean_cons)];
        end
        ret = res;
    end
    
    opt_val = eta_K_opt(det(n), to_min, quant_mean)

    
end

