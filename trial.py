import csv
from Queue import PriorityQueue

"""
Need to rank users(TDID) based on the probability of visiting the "Contact Us Page"
Find P(contact-us page | tdid)
P(tdid | contact-us page) ~= P(contact-us page | tdid) * P(tdid)
"""

reader = csv.reader(open('Programmatic Project_Scoring_TTD pixel fires.csv'), delimiter=' ', quotechar='|')
next(reader)
data_table = []
tdid_dict = {} #contains P(tdid)
contact_dict = {} #contains percentage of tdids that visit contact-us page. P(tdid | contact-us page)

for row in reader:
    data = row[0].split(',')
    tdid = data[0]
    tgid = data[1] #tracking tag id

    if tdid in tdid_dict:
        tdid_dict[tdid] = tdid_dict[tdid] + 1.0
    else:
        tdid_dict[tdid] = 1.0

    value = 0.0
    if tgid == 'qelg9wq':
        value = value + 1.0
    if tdid in contact_dict:
        contact_dict[tdid] = contact_dict[tdid] + value
    else:
        contact_dict[tdid] = value




total = sum(tdid_dict.itervalues(), 0.0)
tdid_dict = {k: v / total for k, v in tdid_dict.iteritems()}

total = sum(contact_dict.itervalues(), 0.0)
contact_dict = {k: v / total for k, v in contact_dict.iteritems()}

pq = PriorityQueue()
max_likelihood = 0.0
for tdid in tdid_dict:
    likelihood = contact_dict[tdid] / tdid_dict[tdid]
    if likelihood > max_likelihood:
        max_likelihood = likelihood

for tdid in tdid_dict:
    likelihood = contact_dict[tdid] / tdid_dict[tdid]
    rank = max_likelihood - likelihood
    pq.put((rank, tdid))

print pq.qsize()
while not pq.empty():
    print pq.get()
