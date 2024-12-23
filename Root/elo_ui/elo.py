# ELO_Scoring_and_UI_from_starburt/elo.py
# ELO: Everyone Loses... Occasionally.

import math

def expected_score(ratingA, ratingB):
    return 1 / (1 + 10 ** ((ratingB - ratingA) / 400))

def update_elo(ratingA, ratingB, winner, K=32):
    EA = expected_score(ratingA, ratingB)
    EB = expected_score(ratingB, ratingA)
    if winner == 'A':
        ratingA_new = ratingA + K * (1 - EA)
        ratingB_new = ratingB + K * (0 - EB)
    else:
        ratingA_new = ratingA + K * (0 - EA)
        ratingB_new = ratingB + K * (1 - EB)
    return ratingA_new, ratingB_new