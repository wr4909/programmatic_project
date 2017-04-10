import pandas as pd

"""
Need to rank users(TDID) based on the probability of visiting the "Contact Us Page"
Find P(contact-us page | tdid)
P(tdid | contact-us page) ~= P(contact-us page | tdid) * P(tdid)
"""
tag_dict = {'Contact Us' : 'qelg9wq', 'Products' : 'pq3g1hn', 'About': 'ipi5afe', 'News': 'yi0fkw5', 'TTD Site': 'wjl3e83'}

df = pd.read_csv('Programmatic Project_Scoring_TTD pixel fires.csv')


def main():
    df['tdid_count'] = df.groupby('tdid')['tdid'].transform('count')
    df['prior'] = df['tdid_count'] * 1.0 / len(df)

    #TODO: posterior: out of all contact page access, what percent were accessed by this tdid
    df['posterior'] = ( df.groupby(['tdid', 'trackingtagid']).size() ) / ( df.loc[df['trackingtagid'] == tag_dict['Contact Us']].size() )
    df['likelihood'] = df['posterior'] / df['prior']
    df.sort('likelihood', ascending=False)
    print df

if __name__ == '__main__':
    main()
