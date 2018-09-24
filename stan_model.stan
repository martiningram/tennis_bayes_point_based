data {
    int<lower=0> num_matches;
    int<lower=0> num_players;
    int<lower=0> num_periods;
    int<lower=0> num_surfaces;
    int<lower=0> num_tournaments;

    int s_id[num_matches]; // server ids
    int r_id[num_matches]; // returner ids
    int period[num_matches]; // the current period
    int surface[num_matches]; // the current surface
    int tournament[num_matches]; // the current tournament

    int spw[num_matches]; // serve points won
    int spt[num_matches]; // serve points total
}
parameters {
    real intercept; // overall intercept
    real<lower=0> sigma_s0; // initial serve ability variation
    real<lower=0> sigma_r0; // initial return ability variation
    real<lower=0> sigma_s; // period variation on serve
    real<lower=0> sigma_r; // period variation on return
    real<lower=0> sigma_surf; // surface ability variation
    real<lower=0> sigma_t; // tournament variation
    vector[num_tournaments] eta_t; // tournament effect
    matrix[num_surfaces, num_players] eta_surface; // surface random component
    matrix[num_periods, num_players] eta_a_s; // random component
    matrix[num_periods, num_players] eta_a_r; // random component
}
transformed parameters {
    matrix[num_periods, num_players] s; // serve abilities
    matrix[num_periods, num_players] r; // return abilities
    matrix[num_surfaces, num_players] surf; // surface abilities
    vector[num_tournaments] t; // tournament offset

    t = sigma_t * eta_t;

    for (i in 1:num_surfaces){
        surf[i, :] = sigma_surf * eta_surface[i, :];
    }

    s[1] = sigma_s0 * eta_a_s[1]; // initial abilities
    r[1] = sigma_r0 * eta_a_r[1]; // initial abilities

    for (p in 2:num_periods){
        s[p] = s[p-1] + sigma_s * eta_a_s[p];
        r[p] = r[p-1] + sigma_r * eta_a_r[p];
    }
}
model {
    vector[num_matches] pred_logits;

    // Priors
    intercept ~ normal(0, 1);
    sigma_s0 ~ normal(0, 1);
    sigma_r0 ~ normal(0, 1);
    sigma_s ~ normal(0, 1);
    sigma_r ~ normal(0, 1);
    sigma_surf ~ normal(0, 1);
    sigma_t ~ normal(0, 1);

    eta_t ~ normal(0, 1);
    to_vector(eta_surface) ~ normal(0, 1);
    to_vector(eta_a_s) ~ normal(0, 1);
    to_vector(eta_a_r) ~ normal(0, 1);

    for (i in 1:num_matches) {
        pred_logits[i] = s[period[i], s_id[i]] - r[period[i], r_id[i]] +
                         surf[surface[i], s_id[i]] - surf[surface[i], r_id[i]] +
                         t[tournament[i]] + intercept;
    }

    spw ~ binomial_logit(spt, pred_logits);

}
generated quantities {
    real tournament_prior;
    real serve_prior;
    real return_prior;
    real surface_prior;
    vector[num_players] prediction_s;
    vector[num_players] prediction_r;

    // Tournament, serve, return and surface priors
    tournament_prior = normal_rng(0, sigma_t);
    serve_prior = normal_rng(0, sigma_s0);
    return_prior = normal_rng(0, sigma_r0);
    surface_prior = normal_rng(0, sigma_surf);

    // Prediction values for r and s -- add variance
    for (i in 1:num_players) {
        prediction_s[i] = normal_rng(s[num_periods, i], sigma_s);
        prediction_r[i] = normal_rng(r[num_periods, i], sigma_r);
    }
}
