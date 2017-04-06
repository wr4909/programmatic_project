import pandas as pd

"""
Need to rank users(TDID) based on the probability of visiting the "Contact Us Page"
"""

broken_df = pd.read_csv('Programmatic Project_Scoring_TTD pixel fires.csv')


def main():
    print type(broken_df)
    print broken_df['country']

if __name__ == '__main__':
    main()
