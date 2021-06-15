
from dataclasses import dataclass
from itertools import groupby


@dataclass
class Ranking:
    target_label:str
    ancillery_agent:str
    ancillery_label:str
    wrong_n:int
    correct_n:int
    wrong_ratio:float
    correct_ratio:float
    wrong_minus_correct_ratio:int
    wrong_rank: float = float('inf')
    correct_rank: float = float('inf')
    wrong_minus_correct_rank: float = float('inf')
    wc_rank: float = float('inf')
    relative_n_rank: float = float('inf')
    relative_rank_rank: float = float('inf')

    def to_mat(self):
        return [self.correct_n, self.wrong_n, self.correct_rank, self.wrong_rank, self.wrong_minus_correct_rank, self.wc_rank, self.relative_n_rank, self.relative_rank_rank]

    def set_correct_rank(self, i):
        self.correct_rank = i

    def set_wrong_rank(self, i):
        self.wrong_rank = i

    def set_wrong_minus_correct_rank(self, i):
        self.wrong_minus_correct_rank = i

    def set_wc_ranking_rank(self):
        self.wc_rank = self.wrong_rank//self.correct_rank

    # [wrong * n_wrong - correct * n_correct]/[n_wrong + n_correct]
    def set_relative_n_rank(self):
        try:
            self.relative_n_rank = (self.wrong_ratio * self.wrong_n - self.correct_ratio * self.correct_n) / (self.wrong_n + self.correct_n)
        except ZeroDivisionError as e:
            self.relative_n_rank = float('inf')

    # [rank wrong/rank correct] * [n_wrong + n_correct]/n_wrong
    def set_relative_rank_rank(self):
        try:
            self.relative_rank_rank = (self.wrong_rank / self.correct_rank * self.wrong_n + self.correct_n) / self.wrong_n
        except ZeroDivisionError as e:
            self.relative_rank_rank = float('inf')

def rank_grouped_rankings(rankings):
    rankings.sort(key=lambda x:x.correct_ratio, reverse=True)
    for i in range(len(rankings)):
        rankings[i].set_correct_rank(i+1)

    rankings.sort(key=lambda x:x.wrong_ratio, reverse=True)
    for i in range(len(rankings)):
        rankings[i].set_wrong_rank(i+1)

    rankings.sort(key=lambda x:x.wrong_minus_correct_ratio, reverse=True)
    for i in range(len(rankings)):
        rankings[i].set_wrong_minus_correct_rank(i+1)

    for i in range(len(rankings)):
        rankings[i].set_wc_ranking_rank()
        rankings[i].set_relative_n_rank()
        rankings[i].set_relative_rank_rank()