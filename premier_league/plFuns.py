import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def evolve(p, q, lmbd, tau, k):
    lmb_plus_tau = lmbd + tau[:, np.newaxis]
    new_p = ((np.exp(-lmb_plus_tau) * (lmb_plus_tau ** k)).T * q).sum(axis=1) * p
    new_p = new_p / new_p.sum()
    new_q = ((np.exp(-lmb_plus_tau) * (lmb_plus_tau ** k)) * p).sum(axis=1) * q
    new_q = new_q / new_q.sum()
    return new_p, new_q


class Team(object):
    def __init__(self, name='team name', lmbd=1, tau=0, goals_scored=0, goals_against=0, points=0):
        self.name = name
        self.lmbd = lmbd
        self.tau = tau
        self.goals_scored = goals_scored
        self.goals_against = goals_against
        self.points = points
        self.goals_scored_scenarios = goals_scored
        self.goals_against_scenarios = goals_against
        self.points_scenarios = points
        self.place_scenarios = 0

        self.lmbd_set = np.linspace(0, 5, 1001)
        self.p = self.lmbd_set * 0 + 1
        self.p = self.p / self.p.sum()
        self.tau_set = np.linspace(0, 5, 1001)
        self.q = self.tau_set * 0 + 1
        self.q = self.q / self.q.sum()

    def simplify(self, threshold=1e-10):
        ind = self.p > threshold
        self.lmbd_set = self.lmbd_set[ind]
        self.p = self.p[ind]
        self.p = self.p / self.p.sum()
        ind = self.q > threshold
        self.tau_set = self.tau_set[ind]
        self.q = self.q[ind]
        self.q = self.q / self.q.sum()

    def __add__(self, other_team, n_scenarios=int(1e4)):
        GH, GA, match_des = self.vs(other_team,n=n_scenarios)
        return np.array([(GH > GA).sum(), (GH == GA).sum(), (GH < GA).sum()]) / n_scenarios

    def vs(self, other_team, n=int(1e4)):
        lH = np.random.choice(self.lmbd_set, size=n, p=self.p) + np.random.choice(other_team.tau_set, size=n,
                                                                                  p=other_team.q)
        gH = np.random.poisson(lH)
        lA = np.random.choice(self.tau_set, size=n, p=self.q) + np.random.choice(other_team.lmbd_set, size=n,
                                                                                 p=other_team.p)
        gA = np.random.poisson(lA)
        match_des = self.name + ' vs ' + other_team.name
        return gH, gA, match_des

    def plt(self):
        plt.plot(self.lmbd_set, self.p, label=self.name + ' lmbda')
        plt.plot(self.tau_set, self.q, label=self.name + ' tau')
        plt.legend()
        plt.grid(True)
        l, t = self.means()
        plt.title('lambda: {:0.2f} tau: {:0.2f}'.format(l, t))

    def means(self):
        return self.p.dot(self.lmbd_set), self.q.dot(self.tau_set)


class Season(object):
    def __init__(self, country):
        urls = pd.Series({'EN': 'http://www.football-data.co.uk/mmz4281/1718/E0.csv',
                          'ES': 'http://www.football-data.co.uk/mmz4281/1718/SP1.csv',
                          'IT': 'http://www.football-data.co.uk/mmz4281/1718/I1.csv',
                          'FR': 'http://www.football-data.co.uk/mmz4281/1718/F1.csv',
                          'DE': 'http://www.football-data.co.uk/mmz4281/1718/D1.csv',
                          'NL': 'http://www.football-data.co.uk/mmz4281/1718/N1.csv'
                          })
        self.url = urls[country[:2]]
        self.all = pd.read_csv(self.url,usecols=['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'])

        self.PointsPerTeam0 = None
        self.Place0 =None
        self.avgPoints0 = None
        self.goals_scored0 = None
        self.goals_against0 =None

        self.PointsPerTeam = np.zeros([1])
        self.Place = np.zeros([1])
        self.avgPoints = np.zeros([1])
        self.goals_scored = np.zeros([1])
        self.goals_against = np.zeros([1])
        self.season_so_far_is_run = False
        self.rest_of_season_is_simulated = False

        self.p_win = np.zeros([1])
        self.p_cl = np.zeros([1])
        self.p_rel = np.zeros([1])

        self.nr_teams = 0
        self.H = None
        self.A = None
        self.Ind = None
        self.team_names = None
        self.Teams = dict()

    def add_match(self,date,home_team,home_goals,away_team,away_goals):
        i=self.all.index.max()+1
        aa=pd.DataFrame({'Date':date,'HomeTeam':home_team,'AwayTeam':away_team,'FTHG':home_goals,'FTAG':away_goals},index=[i])
        self.all=self.all.append(aa)


    def preprocess(self):
        self.all = self.all.dropna()
        H, A, Ind, team_names = GetMainDetails(self.all)
        self.nr_teams = len(team_names)
        self.PointsPerTeam0 = np.zeros([self.nr_teams, 1])
        self.Place0 = np.zeros([self.nr_teams, 1])
        self.avgPoints0 = np.zeros([self.nr_teams, 1])
        self.goals_scored0 = np.zeros([self.nr_teams, 1])
        self.goals_against0 = np.zeros([self.nr_teams, 1])
        self.H = H
        self.A = A
        self.Ind = Ind
        self.team_names = team_names
        for _team in self.team_names:
            self.Teams[_team] = Team(name=_team)


    def calibrate(self, up_to='2018-12-31'):
        self.preprocess()
        up_to = pd.to_datetime(up_to)
        for index, row in self.all.iterrows():
            the_date=pd.to_datetime(row['Date'])
            if the_date <= up_to:
                gH = row['FTHG']
                gA = row['FTAG']
                team1 = self.Teams[row['HomeTeam']]
                team2 = self.Teams[row['AwayTeam']]
                team1.p, team2.q = evolve(team1.p, team2.q, team1.lmbd_set, team2.tau_set, gH)
                team2.p, team1.q = evolve(team2.p, team1.q, team2.lmbd_set, team1.tau_set, gA)
        for team_name in self.Teams:
            team = self.Teams[team_name]
            team.simplify()
            team.lmbd, team.tau = team.means()

    def season_so_far(self):
        for _home in range(self.nr_teams):
            for _away in range(self.nr_teams):
                if self.Ind[_home, _away] > 0:
                    self.goals_scored0[_home, :] = self.goals_scored0[_home, :] + self.H[_home, _away]
                    self.goals_against0[_home, :] = self.goals_against0[_home, :] + self.A[_home, _away]
                    self.goals_scored0[_away, :] = self.goals_scored0[_away, :] + self.A[_home, _away]
                    self.goals_against0[_away, :] = self.goals_against0[_away, :] + self.H[_home, _away]

                    self.Teams[self.team_names[_home]].goals_scored += self.H[_home, _away]
                    self.Teams[self.team_names[_home]].goals_against += self.A[_home, _away]
                    self.Teams[self.team_names[_away]].goals_scored += self.A[_home, _away]
                    self.Teams[self.team_names[_away]].goals_against += self.H[_home, _away]

                    if self.H[_home, _away] > self.A[_home, _away]:
                        self.PointsPerTeam0[_home, :] = self.PointsPerTeam0[_home, :] + 3
                        self.Teams[self.team_names[_home]].points += 3

                    elif self.H[_home, _away] < self.A[_home, _away]:
                        self.PointsPerTeam0[_away, :] = self.PointsPerTeam0[_away, :] + 3
                        self.Teams[self.team_names[_away]].points += 3
                    else:
                        self.PointsPerTeam0[_home, :] = self.PointsPerTeam0[_home, :] + 1
                        self.PointsPerTeam0[_away, :] = self.PointsPerTeam0[_away, :] + 1
                        self.Teams[self.team_names[_home]].points += 1
                        self.Teams[self.team_names[_away]].points += 1

        temp = self.PointsPerTeam0[:, 0].argsort()
        self.Place0[temp, 0] = np.arange(self.nr_teams)
        self.Place0 = self.nr_teams - self.Place0
        self.season_so_far_is_run = True
        self.avgPoints = self.PointsPerTeam0

    def SimulateRestOfSeason(self, nScenarios=int(1e5)):
        if not self.season_so_far_is_run:
            self.season_so_far()
        self.goals_scored = self.goals_scored0 + np.zeros([self.nr_teams, nScenarios])
        self.goals_against = self.goals_against0 + np.zeros([self.nr_teams, nScenarios])
        self.PointsPerTeam = self.PointsPerTeam0 + np.zeros([self.nr_teams, nScenarios])

        for _team in self.Teams:
            self.Teams[_team].goals_scored_scenarios = self.Teams[_team].goals_scored + np.zeros(nScenarios)
            self.Teams[_team].goals_against_scenarios = self.Teams[_team].goals_against + np.zeros(nScenarios)
            self.Teams[_team].points_scenarios = self.Teams[_team].points + np.zeros(nScenarios)

        self.Place = np.zeros([self.nr_teams, nScenarios])
        for _home in range(self.nr_teams):
            for _away in range(self.nr_teams):
                if (self.Ind[_home, _away] == 0) & (_home != _away):
                    _home_name = self.team_names[_home]
                    _away_name = self.team_names[_away]
                    GH, GA, _ = self.Teams[_home_name].vs(self.Teams[_away_name],n=nScenarios)
                    self.goals_scored[_home, :] += GH
                    self.goals_against[_home, :] += GA
                    self.goals_scored[_away, :] += GA
                    self.goals_against[_away, :] += GH

                    self.Teams[_home_name].goals_scored_scenarios += GH
                    self.Teams[_home_name].goals_against_scenarios += GA
                    self.Teams[_away_name].goals_scored_scenarios += GA
                    self.Teams[_away_name].goals_against_scenarios += GH

                    indHomeWins = GH > GA
                    self.PointsPerTeam[_home, indHomeWins] += 3
                    self.Teams[_home_name].points_scenarios[indHomeWins] += 3
                    indAwayWins = GA > GH
                    self.PointsPerTeam[_away, indAwayWins] += 3
                    self.Teams[_away_name].points_scenarios[indAwayWins] += 3
                    indDraw = (GA == GH)
                    self.PointsPerTeam[_home, indDraw] += 1
                    self.PointsPerTeam[_away, indDraw] += 1
                    self.Teams[_home_name].points_scenarios[indDraw] += 1
                    self.Teams[_away_name].points_scenarios[indDraw] += 1
        dgd = self.goals_scored - self.goals_against
        dgd = dgd - dgd.min()
        dgd = 0.1 * dgd / dgd.max()
        dg = self.goals_scored
        dg = dg - dg.min()
        dg = 0.01 * dg / dg.max()
        ordering = self.PointsPerTeam + dgd + dg
        for i in range(nScenarios):
            temp = ordering[:, i].argsort()
            self.Place[temp, i] = np.arange(self.nr_teams)
        self.Place = self.nr_teams - self.Place
        for _i in range(self.nr_teams):
            _team = self.team_names[_i]
            self.Teams[_team].place_scenarios = self.Place[_i, :]
        self.rest_of_season_is_simulated = True
        self.avgPoints = self.PointsPerTeam.mean(axis=1)
        self.p_win = (self.Place == 1).sum(axis=1) / nScenarios
        self.p_cl = (self.Place <= 4).sum(axis=1) / nScenarios
        self.p_rel = (self.Place >= 17).sum(axis=1) / nScenarios

    def get_all_results(self):
        lmbd = np.array([self.Teams[x].lmbd for x in self.team_names])
        tau = np.array([self.Teams[x].tau for x in self.team_names])

        df = pd.DataFrame(
            {'lambda': lmbd.round(3), 'tau': tau.round(3), 'Points': self.PointsPerTeam0[:, 0].astype(int),
             'GF': self.goals_scored0[:, 0].astype(int), 'GA': self.goals_against0[:, 0].astype(int),
             'Average Points': self.avgPoints.round(2), 'Win': 100 * self.p_win,'Relegated': 100 * self.p_rel,
             'CL': 100 * self.p_cl, 'Average Goals Scored': self.goals_scored.mean(axis=1).round(2),
             'Average Goals Against': self.goals_against.mean(axis=1).round(2)}, index=self.team_names)
        df = df.sort_values(by='Average Points', ascending=False)
        return df

    def conf_int_plot(self, ci=95, kind='points'):
        p1 = (100 + ci) / 2
        p0 = (100 - ci) / 2
        if kind == 'points':
            the_var = self.PointsPerTeam
            yticks = np.arange(0, self.PointsPerTeam.max() + 1, 5)
        else:
            the_var = self.Place
            yticks = np.arange(self.nr_teams) + 1

        y = np.percentile(the_var, [50], axis=1)[0, :]
        thelabels = np.array(self.team_names)
        loc = y.argsort()
        dy = np.percentile(the_var, [p0, p1], axis=1)
        dy = dy[:, loc]
        y = y[loc]
        thelabels = thelabels[loc]
        x = np.arange(self.nr_teams)
        plt.errorbar(x, y, yerr=[y - dy[0, :], dy[1, :] - y])
        plt.grid(True)
        plt.xticks(x, thelabels, rotation='vertical');
        plt.yticks(yticks);
        fig = plt.gcf()
        fig.set_size_inches(20, 12)

def GetMainDetails(data, upto=pd.datetime(2115, 8, 9)):
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%y')
    data = data.loc[data['Date'] <= upto]
    team_names = np.unique([data['HomeTeam'].values, data['AwayTeam'].values])
    NrTeams = team_names.shape[0]
    AwayTeam = []
    HomeTeam = []
    GH = []
    GA = []
    # print(data['FTHG'].sum()/data['FTAG'].sum())
    H = np.zeros([NrTeams, NrTeams])
    A = np.zeros([NrTeams, NrTeams])
    Ind = np.zeros([NrTeams, NrTeams])
    for index, row in data.iterrows():
        _HomeTeam = int(np.where(team_names == row['HomeTeam'])[0])
        _AwayTeam = int(np.where(team_names == row['AwayTeam'])[0])
        if not np.isnan(row['FTHG']):
            HomeTeam.append(_HomeTeam)
            AwayTeam.append(_AwayTeam)
            GH.append(row['FTHG'])
            GA.append(row['FTAG'])
            H[_HomeTeam, _AwayTeam] = row['FTHG']
            A[_HomeTeam, _AwayTeam] = row['FTAG']
            Ind[_HomeTeam, _AwayTeam] = 1
    H = np.matrix(H)
    A = np.matrix(A)
    Ind = np.matrix(Ind)
    return H, A, Ind, list(team_names)
