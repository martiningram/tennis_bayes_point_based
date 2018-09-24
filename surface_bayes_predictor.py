import os
import pystan
import numpy as np
import pandas as pd
import cPickle as pkl
from dateutil.relativedelta import relativedelta
from scipy.special import expit
from tpr.iid.winning_prob import prob_win_match_a
from sklearn.preprocessing import LabelEncoder
from tpr.models.predicted_match import PredictedMatch
from tpr.models.match_predictor import MatchPredictor, CannotCalculateException


class SurfaceBayesPredictor(MatchPredictor):

    def __init__(self, dataset, start_date, period_length=3, use_cache=True,
                 store_posterior_dir=None, quick_test_run=False):

        super(SurfaceBayesPredictor, self).__init__(dataset)

        self.period_length = period_length
        self.start_date = start_date
        self.reduced_data = self.reduce_to_relevant_data(dataset.full_df)
        self.store_posterior_dir = store_posterior_dir
        self.quick_test_run = quick_test_run

        pickle_name = 'tpr/models/stan_models/surface_non_centred.pkl'

        if os.path.isfile(pickle_name) and use_cache:

            self.model = pkl.load(open(pickle_name, 'rb'))

        else:

            assert(os.path.isfile(
                'tpr/models/stan_models/surface_non_centred.stan'))

            self.model = pystan.StanModel(
                file='tpr/models/stan_models/surface_non_centred.stan')

            pkl.dump(self.model, open(pickle_name, 'wb'))

        # This will be a dict mapping period --> predictions
        self.cached_results = dict()

    @staticmethod
    def transform_if_present(class_name, encoder):
        if class_name in encoder.classes_:
            return encoder.transform([class_name])[0]
        else:
            return None

    @staticmethod
    def diff_month(d1, d2):
        # Credit to:
        # https://stackoverflow.com/questions/4039879/best-way-to-find-the-months-between-two-dates
        return (d1.year - d2.year) * 12 + d1.month - d2.month

    def calculate_period(self, date):
        # This function expects datetime.date objects
        # TODO: Replace this with the Glicko period calculation to avoid
        # duplication

        months_since = self.diff_month(date, self.start_date)

        return months_since // self.period_length + 1

    def period_to_date(self, period):

        return self.start_date + relativedelta(
            months=self.period_length * (period - 1))

    @staticmethod
    def calculate_spw_dist(p1_id, p2_id, surface_id, tournament_id,
                           posteriors):
        # Assume posteriors has no more time dependence, but has only one
        # column corresponding to the current epoch

        p1_serve_skill = (posteriors['prior_s'] if p1_id is None else
                          posteriors['s'][:, p1_id])
        p2_serve_skill = (posteriors['prior_s'] if p2_id is None else
                          posteriors['s'][:, p2_id])
        p1_return_skill = (posteriors['prior_r'] if p1_id is None else
                           posteriors['r'][:, p1_id])
        p2_return_skill = (posteriors['prior_r'] if p2_id is None else
                           posteriors['r'][:, p2_id])
        p1_surface_skill = (posteriors['prior_surf']
                            if surface_id is None or p1_id is None
                            else posteriors['surf'][:, surface_id, p1_id])
        p2_surface_skill = (posteriors['prior_surf']
                            if surface_id is None or p2_id is None
                            else posteriors['surf'][:, surface_id, p2_id])
        tournament_intercept = (posteriors['prior_t'] if tournament_id is None
                                else posteriors['t'][:, tournament_id])

        # Calculate the expected performance of each player
        p1_spw = expit(p1_serve_skill - p2_return_skill
                       + p1_surface_skill
                       - p2_surface_skill
                       + tournament_intercept
                       + posteriors['intercept'])

        p2_spw = expit(p2_serve_skill - p1_return_skill
                       + p2_surface_skill
                       - p1_surface_skill
                       + tournament_intercept
                       + posteriors['intercept'])

        return p1_spw, p2_spw

    def fit_model(self, fit_period):

        # Subset the data
        relevant_data = self.reduced_data[self.reduced_data['period'] <
                                          fit_period]

        surface_encoder = LabelEncoder()
        player_encoder = LabelEncoder()
        tournament_encoder = LabelEncoder()

        surface_ids = surface_encoder.fit_transform(
            relevant_data['surface']) + 1

        tournament_ids = tournament_encoder.fit_transform(
            relevant_data['tournament']) + 1

        players = (relevant_data['server'].tolist() +
                   relevant_data['returner'].tolist())

        player_encoder.fit(players)

        server_ids = player_encoder.transform(
            relevant_data['server'].values) + 1
        returner_ids = player_encoder.transform(
            relevant_data['returner'].values) + 1

        spw = relevant_data['sp_won'].values.astype(int)
        spt = relevant_data['sp_total'].values.astype(int)
        period = relevant_data['period']

        num_periods = len(relevant_data['period'].unique())
        num_surfaces = len(surface_encoder.classes_)

        model_data = {'num_matches': relevant_data.shape[0],
                      'num_players': len(player_encoder.classes_),
                      's_id': server_ids,
                      'r_id': returner_ids,
                      'spw': spw,
                      'spt': spt,
                      'period': period,
                      'num_periods': num_periods,
                      'surface': surface_ids,
                      'num_surfaces': num_surfaces,
                      'tournament': tournament_ids,
                      'num_tournaments': len(tournament_encoder.classes_)}

        print('Sampling from the posterior...')
        if self.quick_test_run:
            posteriors = self.model.sampling(data=model_data, iter=100,
                                             chains=1, algorithm='HMC', warmup=0)
        else:
            posteriors = self.model.sampling(data=model_data)

        model_info = str(posteriors)

        print('Extracting posterior samples...')
        posteriors = posteriors.extract()

        if self.store_posterior_dir is not None:

            print('Summarising posteriors...')

            posterior_summaries = self.summarise_posteriors(
                posteriors, player_encoder, surface_encoder,
                tournament_encoder)

            # Also add the fit string with the r hats and so on
            posterior_summaries['fit_str'] = model_info

            target_file = os.path.join(
                self.store_posterior_dir, '{}_{}_months_{}.pkl'.format(
                    self.start_date.isoformat(),
                    self.period_length,
                    self.period_to_date(fit_period).isoformat()))

            print('Saving posterior samples...')
            pkl.dump(posterior_summaries, open(target_file, 'wb'))

        # Subset to only the latest ones
        posteriors = {'r': posteriors['prediction_r'],
                      's': posteriors['prediction_s'],
                      'surf': posteriors['surf'],
                      't': posteriors['t'],
                      'intercept': posteriors['intercept'],
                      'prior_r': posteriors['return_prior'],
                      'prior_s': posteriors['serve_prior'],
                      'prior_surf': posteriors['surface_prior'],
                      'prior_t': posteriors['tournament_prior']}

        results = {'surface_encoder': surface_encoder,
                   'player_encoder': player_encoder,
                   'tournament_encoder': tournament_encoder,
                   'posteriors': posteriors}

        return results

    def summarise_posteriors(self, posteriors, player_encoder, surface_encoder,
                             tournament_encoder):

        # Store return skills
        # Target: a dictionary mapping date to a df with columns of players and
        # rows of samples
        r_summaries = dict()
        s_summaries = dict()

        num_periods = posteriors['r'].shape[1]

        player_names = player_encoder.classes_
        surface_names = surface_encoder.classes_

        for cur_period in range(num_periods):

            # Need to add 1 because periods start with one
            cur_date = self.period_to_date(cur_period + 1)

            cur_s = posteriors['s'][:, cur_period, :]
            cur_r = posteriors['r'][:, cur_period, :]

            cur_s_df = pd.DataFrame(cur_s, columns=player_names)
            cur_r_df = pd.DataFrame(cur_r, columns=player_names)

            r_summaries[cur_date] = cur_r_df
            s_summaries[cur_date] = cur_s_df

        # Add the predictions
        r_summaries[self.period_to_date(num_periods + 1)] = pd.DataFrame(
            posteriors['prediction_r'], columns=player_names)
        s_summaries[self.period_to_date(num_periods + 1)] = pd.DataFrame(
            posteriors['prediction_s'], columns=player_names)

        # Next, summarise the surface skills
        # Target: Dict mapping surface to DataFrame of samples
        surface_summaries = dict()

        for i, cur_surface_name in enumerate(surface_names):

            cur_posterior = posteriors['surf'][:, i, :]

            surface_summaries[cur_surface_name] = pd.DataFrame(
                cur_posterior, columns=player_names)

        # Next: tournament summaries
        # Target: DataFrame of samples per tournament (columns)
        tournament_summaries = pd.DataFrame(
            posteriors['t'], columns=tournament_encoder.classes_)

        # Store all other posteriors in DataFrame
        other_quantities = set(posteriors.keys()) - set(
            ['r', 's', 't', 'surf', 'log_lik'])

        other_quantities = [x for x in other_quantities if 'eta'
                            not in x and 'prediction' not in x]

        results = dict()

        for cur_quantity in other_quantities:

            cur_posterior = posteriors[cur_quantity]
            results[cur_quantity] = cur_posterior

        results = pd.DataFrame.from_dict(results)

        model_information = {'start_date': self.start_date,
                             'last_period': max(r_summaries.keys()),
                             'period_length': self.period_length}

        # Combine all these into dictionary
        model_summary = {'posteriors': {'r': r_summaries,
                                        's': s_summaries,
                                        'surface': surface_summaries,
                                        'tournaments': tournament_summaries,
                                        'others': results},
                         'model_info': model_information}

        return model_summary

    def predict_match(self, match):

        cur_date = match.date
        cur_period = self.calculate_period(cur_date)
        assert(cur_period > 1)

        # Find the ids
        if cur_period in self.cached_results:

            model_results = self.cached_results[cur_period]

        else:

            model_results = self.fit_model(cur_period)
            self.cached_results[cur_period] = model_results

        player_encoder = model_results['player_encoder']
        surface_encoder = model_results['surface_encoder']
        tournament_encoder = model_results['tournament_encoder']
        posteriors = model_results['posteriors']

        p1_id = self.transform_if_present(match.p1, player_encoder)
        p2_id = self.transform_if_present(match.p2, player_encoder)
        surface_id = self.transform_if_present(match.surface, surface_encoder)
        tournament_id = self.transform_if_present(
            match.tournament_name, tournament_encoder)

        p1_spw_dist, p2_spw_dist = self.calculate_spw_dist(
            p1_id, p2_id, surface_id, tournament_id, posteriors)

        # TODO: Make 100% sure this is right
        results = prob_win_match_a(p1_spw_dist, p2_spw_dist,
                                   best_of_five=match.bo5)

        uncertainty = np.std(results)

        model_details = {'model': 'Surface Bayes',
                         'match_period': cur_period,
                         'fit_start': self.start_date,
                         'fit_every': self.period_length,
                         'spw_stds': {match.p1: p1_spw_dist.std(),
                                      match.p2: p2_spw_dist.std()},
                         'p1_win_prob_std': uncertainty,
                         'p1_unknown': p1_id is None,
                         'p2_unknown': p2_id is None,
                         'surface_unknown': surface_id is None,
                         'tournament_unknown': tournament_id is None}

        p1_win_prob = np.mean(results)

        predicted_match = PredictedMatch.from_match(
            match, win_probs={match.p1: p1_win_prob,
                              match.p2: 1 - p1_win_prob},
            spw_predictions={match.p1: p1_spw_dist.mean(),
                             match.p2: p2_spw_dist.mean()},
            model_details=model_details)

        return predicted_match

    def reduce_to_relevant_data(self, full_df):

        # Subset full df to relevant date range
        full_df = full_df[full_df['start_date'] > self.start_date]

        # Calculate serve points won
        loser_spw_won = (full_df['winner_return_points_total'] -
                         full_df['winner_return_points_won'])

        loser_sp_total = full_df['winner_return_points_total']

        losers = full_df['loser']

        winner_spw_won = (full_df['loser_return_points_total'] -
                          full_df['loser_return_points_won'])

        winner_spw_total = full_df['loser_return_points_total']

        winners = full_df['winner']

        # This assumes pandas datetimes
        months_since_start = ((full_df['start_date'] -
                               self.start_date) / np.timedelta64(
                                   1, 'M')).astype(int)

        period = (months_since_start // self.period_length) + 1

        surface = full_df['surface']
        tournament = full_df['tournament_name']

        winner_frame = pd.DataFrame({'server': winners, 'returner': losers,
                                     'sp_won': winner_spw_won, 'sp_total':
                                     winner_spw_total, 'period': period,
                                     'surface': surface, 'tournament':
                                     tournament})

        loser_frame = pd.DataFrame({'server': losers, 'returner': winners,
                                    'sp_won': loser_spw_won, 'sp_total':
                                    loser_sp_total, 'period': period,
                                    'surface': surface, 'tournament':
                                    tournament})

        combined = pd.concat([winner_frame, loser_frame])

        # Drop the index
        combined = combined.reset_index(drop=True)

        return combined
