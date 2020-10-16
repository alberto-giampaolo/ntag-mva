import get_ps


effruns = []
effile = '/disk02/usr6/giampaol/ntag-mva/models/overtrain/eff_runs.csv'
with open(effile, 'r') as fl:
    runs = fl.readline()
    runs = runs.strip('#').strip().split(' | ')
    effruns = [int(float(r)) for r in runs]

bgruns = []
bgfile = '/disk02/usr6/giampaol/ntag-mva/models/overtrain/bg_runs.csv'
with open(bgfile, 'r') as fl:
    runs = fl.readline()
    runs = runs.strip('#').strip().split(' | ')
    runs = [n.split('-') for n in runs]
    bgruns = [[int(float(n)) for n in r] for r in runs]

# with open('/disk02/usr6/giampaol/ntag-mva/models/overtrain/ps_eff.txt', 'w') as fl:

#     for r in effruns:
#         print("Getting preselection for run %d" % r)
#         print(get_ps.get_ps_eff(r))
#     fl.writelines()

for r in effruns:
    print("Getting preselection for run %d" % r)
    print(get_ps.get_ps_eff(r))

# for r1, r2 in bgruns:
#     print("Getting preselection for rus %d-%d" % (r1, r2))
#     print(get_ps.get_ps_bg(r1, r2))
