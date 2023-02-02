function [ beta, ehat, J ] = train_nonlinear_map( X, Y )
    model = 'logistic5';
    warning ('off','all');

    ymax = max(Y);
    ymin = min(Y);

    beta0(1) = ymax;
    beta0(2) = ymin;
    beta0(3) = mean(X);
    beta0(4) = 0.5;

    [beta, ehat, J] = nlinfit(X, Y, model, beta0);
end

