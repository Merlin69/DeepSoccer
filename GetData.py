import csv
import datetime
import ToolBox
import random


class Data:
    def __init__(self, filename, nb_times):
        self.filename = filename
        self.country_to_id = {}
        self.id_to_country = []
        self.nb_teams = 159
        self.nb_times = 14
        self.train = {}
        self.test = {}
        self.elo = None
        self.metadata = {'nb_teams': self.nb_teams, 'nb_times': self.nb_times}

    def get(self, p):
        draw_count = 0.
        match_count = 0.
        self.train = dict(dict(time=[], team_h=[], team_a=[], res=[], score_h=[], score_a=[]), **self.metadata)
        self.test = dict(dict(time=[], team_h=[], team_a=[], res=[], score_h=[], score_a=[]), **self.metadata)
        with open(self.filename, 'rb') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                if row:
                    [score1, score2, year, day, id1, id2] = row
                    if score1 == score2:
                        draw_count += 1.
                    match_count += 1.
                    date = int(year) - 2003
                    if random.random() < float(p):
                        proxy = self.train
                    else:
                        proxy = self.test
                    self.append_match(proxy, int(id1), int(id2), int(score1), int(score2), date)
        print "LOOOK AT ME LOL :::::: ", draw_count/match_count

    def create_matches(self, matches):
        formated_matches = dict(dict(time=[], team_h=[], team_a=[], res=[], score_h=[], score_a=[]), **self.metadata)
        for team_h, team_a, date in matches:
            if date == 'last':
                date = self.nb_times - 1
            self.append_match(formated_matches, team_h, team_a, 0, 0, date)
        return formated_matches

    def append_match(self, proxy, id1, id2, score1, score2, date):
        proxy['time'].append(ToolBox.make_vector(date, self.nb_times))
        proxy['team_h'].append(ToolBox.make_vector(id1, self.nb_teams))
        proxy['team_a'].append(ToolBox.make_vector(id2, self.nb_teams))
        score_team_h = min(int(score1), 9)
        score_team_a = min(int(score2), 9)
        proxy['score_h'].append(ToolBox.make_vector(score_team_h, 10))
        proxy['score_a'].append(ToolBox.make_vector(score_team_a, 10))
        proxy['res'].append(ToolBox.result_vect(int(score1) - int(score2)))

    def set_elos(self, elo):
        self.elo = elo

    def get_elos(self, countries=None, times='all'):
        if countries is None:
            countries = [self.id_to_country[i] for i in range(self.nb_teams)]
        else:
            countries = map(ToolBox.format_name, countries)
        if times == 'all':
            times = range(self.nb_times)
        elif times == 'last':
            times = [self.nb_times - 1]
        elos = {}
        for country in countries:
            elos[country] = [200*self.elo[self.country_to_id[country]][t] for t in times]
        return elos
