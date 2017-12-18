import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Team(object):
    def __init__(self,name='team name',lmbd=1,tau=0,goals_scored=0,goals_against=0,points=0):
        self.name=name
        self.lmbd=lmbd
        self.tau=tau
        self.goals_scored=goals_scored
        self.goals_against = goals_against
        self.points = points
        self.goals_scored_scenarios = goals_scored
        self.goals_against_scenarios = goals_against
        self.points_scenarios = points
        self.place_scenarios = 0

    def versus(self,other_team,n_scenarios=int(1e4)):
        lH=self.lmbd+other_team.tau
        lA=other_team.lmbd+self.tau
        GH=np.random.poisson(lH, n_scenarios)
        GA=np.random.poisson(lA, n_scenarios)
        return np.array([(GH>GA).sum() ,(GH==GA).sum() ,(GH<GA).sum()])/n_scenarios
    def __add__(self,other_team,n_scenarios=int(1e4)):
        lH=self.lmbd+other_team.tau
        lA=other_team.lmbd+self.tau
        GH=np.random.poisson(lH, n_scenarios)
        GA=np.random.poisson(lA, n_scenarios)
        return np.array([(GH>GA).sum() ,(GH==GA).sum() ,(GH<GA).sum()])/n_scenarios
    

class Season(object):
    def __init__(self, country):
        urls = pd.Series({'EN': 'http://www.football-data.co.uk/mmz4281/1718/E0.csv',
                       'ES': 'http://www.football-data.co.uk/mmz4281/1718/SP1.csv',
                       'IT': 'http://www.football-data.co.uk/mmz4281/1718/I1.csv',
                       'FR': 'http://www.football-data.co.uk/mmz4281/1718/F1.csv',
                       'DE': 'http://www.football-data.co.uk/mmz4281/1718/D1.csv',
                       'NL': 'http://www.football-data.co.uk/mmz4281/1718/N1.csv'
                       })
        self.url = urls[country]
        self.all = pd.read_csv(self.url )
        H, A, Ind, team_names = GetMainDetails(self.all)
        self.nr_teams = len(team_names)
        self.H = H
        self.A = A
        self.Ind = Ind
        self.team_names = team_names
        #self.lambd = np.ones(len(self.team_names))
        #self.tau = np.ones(len(self.team_names)) / 10
        self.PointsPerTeam0 = np.zeros([self.nr_teams, 1])
        self.Place0 = np.zeros([self.nr_teams, 1])
        self.avgPoints0 = np.zeros([self.nr_teams, 1])
        self.goals_scored0 = np.zeros([self.nr_teams, 1])
        self.goals_against0 = np.zeros([self.nr_teams, 1])

        self.PointsPerTeam = np.zeros([1])
        self.Place = np.zeros([1])
        self.avgPoints = np.zeros([1])
        self.goals_scored = np.zeros([1])
        self.goals_against = np.zeros([1])
        self.season_so_far_is_run=False
        self.rest_of_season_is_simulated = False

        self.p_win= np.zeros([1])
        self.p_cl= np.zeros([1])
        self.Teams=dict()
        for _team in self.team_names:
            self.Teams[_team]=Team(name=_team)

    def calibrate(self, eta=1 / 100):
        lambd = np.ones(len(self.team_names))
        tau = np.ones(len(self.team_names)) / 10
        f, df_l, df_t = LogLikelihood(lambd, tau, self.H, self.A, self.Ind)
        df2 = (df_l ** 2 + df_t ** 2).sum()
        while df2 > 1e-3:
            lambd += eta * df_l
            tau += eta * df_t
            f, df_l, df_t = LogLikelihood(lambd, tau, self.H, self.A, self.Ind)
            df2 = (df_l ** 2 + df_t ** 2).sum()
        #self.lambd = lambd ** 2
        #self.tau = tau ** 2
        for _i in range(len(self.team_names)):
            _team=self.team_names[_i]
            self.Teams[_team].lmbd=lambd[_i]**2
            self.Teams[_team].tau=tau[_i]**2

    def season_so_far(self):
        for _home in range(self.nr_teams):
            for _away in range(self.nr_teams):
                if self.Ind[_home, _away] > 0:
                    self.goals_scored0[_home, :] = self.goals_scored0[_home, :] + self.H[_home, _away]
                    self.goals_against0[_home, :] = self.goals_against0[_home, :] + self.A[_home, _away]
                    self.goals_scored0[_away, :] = self.goals_scored0[_away, :] + self.A[_home, _away]
                    self.goals_against0[_away, :] = self.goals_against0[_away, :] + self.H[_home, _away]

                    self.Teams[self.team_names[_home]].goals_scored+=self.H[_home, _away]
                    self.Teams[self.team_names[_home]].goals_against +=self.A[_home, _away]
                    self.Teams[self.team_names[_away]].goals_scored += self.A[_home, _away]
                    self.Teams[self.team_names[_away]].goals_against+= self.H[_home, _away]

                    if self.H[_home, _away] > self.A[_home, _away]:
                        self.PointsPerTeam0[_home, :] = self.PointsPerTeam0[_home, :] + 3
                        self.Teams[self.team_names[_home]].points+=3

                    elif self.H[_home, _away] < self.A[_home, _away]:
                        self.PointsPerTeam0[_away, :] = self.PointsPerTeam0[_away, :] + 3
                        self.Teams[self.team_names[_away]].points += 3
                    else:
                        self.PointsPerTeam0[_home, :] = self.PointsPerTeam0[_home, :] + 1
                        self.PointsPerTeam0[_away, :] = self.PointsPerTeam0[_away, :] + 1
                        self.Teams[self.team_names[_home]].points += 1
                        self.Teams[self.team_names[_away]].points += 1

        temp = self.PointsPerTeam0[:, 0].argsort()
        self.Place0[temp,0] = np.arange(self.nr_teams)
        self.Place0 = self.nr_teams - self.Place0
        self.season_so_far_is_run = True
        self.avgPoints=self.PointsPerTeam0

    def SimulateRestOfSeason(self, nScenarios=int(1e5)):
        if not self.season_so_far_is_run:
            self.season_so_far()
        self.goals_scored = self.goals_scored0 + np.zeros([self.nr_teams, nScenarios])
        self.goals_against = self.goals_against0 + np.zeros([self.nr_teams, nScenarios])
        self.PointsPerTeam = self.PointsPerTeam0 + np.zeros([self.nr_teams, nScenarios])

        for _team in self.Teams:
            self.Teams[_team].goals_scored_scenarios=self.Teams[_team].goals_scored+np.zeros(nScenarios)
            self.Teams[_team].goals_against_scenarios = self.Teams[_team].goals_against + np.zeros(nScenarios)
            self.Teams[_team].points_scenarios = self.Teams[_team].points + np.zeros(nScenarios)

        self.Place = np.zeros([self.nr_teams, nScenarios])
        for _home in range(self.nr_teams):
            for _away in range(self.nr_teams):
                if (self.Ind[_home, _away] == 0) & (_home != _away):

                    _home_name=self.team_names[_home]
                    _away_name = self.team_names[_away]
                    lH=self.Teams[_home_name].lmbd+self.Teams[_away_name].tau
                    lA = self.Teams[_home_name].tau + self.Teams[_away_name].lmbd
                    #lH = self.lambd[_home] + self.tau[_away]
                    #lA = self.lambd[_away] + self.tau[_home]
                    GH = np.random.poisson(lH, nScenarios)
                    GA = np.random.poisson(lA, nScenarios)
                    self.goals_scored[_home, :] +=GH
                    self.goals_against[_home, :] += GA
                    self.goals_scored[_away, :] += GA
                    self.goals_against[_away, :] += GH

                    self.Teams[_home_name].goals_scored_scenarios+=GH
                    self.Teams[_home_name].goals_against_scenarios += GA
                    self.Teams[_away_name].goals_scored_scenarios += GA
                    self.Teams[_away_name].goals_against_scenarios += GH

                    indHomeWins = GH > GA
                    self.PointsPerTeam[_home, indHomeWins] += 3
                    self.Teams[_home_name].points_scenarios[indHomeWins]+=3
                    indAwayWins = GA > GH
                    self.PointsPerTeam[_away, indAwayWins] += 3
                    self.Teams[_away_name].points_scenarios[indAwayWins] += 3
                    indDraw = (GA == GH)
                    self.PointsPerTeam[_home, indDraw] += 1
                    self.PointsPerTeam[_away, indDraw] += 1
                    self.Teams[_home_name].points_scenarios[indDraw] += 1
                    self.Teams[_away_name].points_scenarios[indDraw] += 1
        dgd=self.goals_scored-self.goals_against
        dgd=dgd-dgd.min()
        dgd=0.1*dgd/dgd.max()
        dg = self.goals_scored
        dg = dg - dg.min()
        dg = 0.01 * dg / dg.max()
        ordering=self.PointsPerTeam+dgd+dg
        for i in range(nScenarios):
            temp = ordering[:, i].argsort()
            self.Place[temp, i] = np.arange(self.nr_teams)
        self.Place = self.nr_teams - self.Place
        for _i in range(self.nr_teams):
            _team=self.team_names[_i]
            self.Teams[_team].place_scenarios=self.Place[_i,:]
        self.rest_of_season_is_simulated = True
        self.avgPoints = self.PointsPerTeam.mean(axis=1)
        self.p_win=(self.Place==1).sum(axis=1)/nScenarios
        self.p_cl = (self.Place <= 4).sum(axis=1) / nScenarios
    def get_all_results(self):
        lmbd=np.array([self.Teams[x].lmbd for x in self.team_names])
        tau =np.array( [self.Teams[x].tau for x in self.team_names])


        df = pd.DataFrame({'lambda': lmbd.round(3), 'tau': tau.round(3), 'Points': self.PointsPerTeam0[:, 0].astype(int),
                           'GF': self.goals_scored0[:, 0].astype(int), 'GA': self.goals_against0[:, 0].astype(int),
                           'Average Points': self.avgPoints.round(2), 'Win': 100 * self.p_win,
                           'CL': 100 * self.p_cl,'Average Goals Scored':self.goals_scored.mean(axis=1).round(2),
                           'Average Goals Against': self.goals_against.mean(axis=1).round(2)}, index=self.team_names)
        df=df.sort_values(by='Average Points', ascending=False)
        return df
    def conf_int_plot(self,ci=95,kind='points'):
        p1=(100+ci)/2
        p0=(100-ci)/2
        if kind=='points':
            the_var=self.PointsPerTeam
            yticks = np.arange(0,self.PointsPerTeam.max()+1,5)
        else:
            the_var = self.Place
            yticks=np.arange(self.nr_teams)+1

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

    def save_results(self, fname='multipage.pdf'):
        if not self.rest_of_season_is_simulated:
            self.goals_scored = self.goals_scored0 + np.zeros([self.nr_teams, 10])
            self.goals_against = self.goals_against0 + np.zeros([self.nr_teams, 10])
            self.PointsPerTeam = self.PointsPerTeam0 + np.zeros([self.nr_teams, 10])
            self.Place=self.Place0+np.zeros([self.nr_teams, 10])
        from matplotlib.backends.backend_pdf import PdfPages
        loc = np.argsort(-self.PointsPerTeam.mean(axis=1))
        _avgPoints = -np.sort(-self.PointsPerTeam.mean(axis=1))
        _Place = self.Place[loc, :]
        _Teams = np.array(self.team_names)[loc]
        _PointsPerTeam = self.PointsPerTeam[loc, :]
        _goals_scored = self.goals_scored[loc, :]
        _goals_against = self.goals_against[loc, :]

        plt.rcParams['figure.figsize'] = [20, 12]
        pp = PdfPages(fname)
        for t in range(self.nr_teams):
            plt.figure(t)
            f, axarr = plt.subplots(1, 2)
            a = np.histogram(_Place[t, :], bins=np.linspace(0.5, self.nr_teams + 0.5, 4 * self.nr_teams + 1))
            axarr[0].bar(a[1][1:] - 0.25, 100 * a[0] / a[0].sum())
            axarr[0].grid(True)
            axarr[0].set_xticks(np.arange(self.nr_teams) + 1)
            axarr[0].set_title(_Teams[t] + ' ' + str(_avgPoints[t]))

            a = np.histogram(_PointsPerTeam[t, :])
            axarr[1].bar(a[1][1:] - 0.25, 100 * a[0] / a[0].sum())
            axarr[1].grid(True)
            axarr[1].set_title(_Teams[t] + ' ' + str(_avgPoints[t]))

            plt.savefig(pp, format='pdf')
        pp.close()


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


def LogLikelihood(lambd, tau, H, A, Ind):
    M1 = Ind + Ind.transpose()
    M2 = np.multiply(Ind, H) + 1.0 * np.multiply(Ind.transpose(), A.transpose())
    lpt = (lambd ** 2)[:, np.newaxis] + (tau ** 2)
    f = -np.trace(lpt * M1) + np.trace(np.log(lpt) * M2)
    M3 = -M1 + np.divide(M2, lpt)
    df_l = np.asarray(M3.sum(axis=1)).reshape(-1) * lambd
    df_t = np.asarray(M3.sum(axis=0)).reshape(-1) * tau
    return f, df_l, df_t
