from node_normal import *
from node_invgamma import *
from network import *
from operator import itemgetter
import numpy

logging.basicConfig(level=logging.WARNING, format='[%(levelname)s] %(module)s %(funcName)s(): %(message)s')

"""
The nodes of interest are the golfermeans. It is insightful to compare
the posterior distributions for different golfers. We took the 5th, 50th,
and 95th percentiles of samples in the posterior distribution for each
golfer and printed these out in ranked order. In Python, this looked like:

ability = []
for golfer in golfermean:
    samples = golfermean[golfer].samples[:]
    samples.sort()
    median = samples[nsamples//2]
    low = samples[int(.05 * nsamples)]
    high = samples[int(.95 * nsamples)]
    ability.append( (golfer, low, median, high) )

ability.sort(lambda x, y: cmp(x[2], y[2]) )
i=1
for golfer, low, median, high in ability:
    print '%d: %s %f; 90%% interval: (%f, %f)' % (i, golfer, median, low, high)
    i += 1


The final results looked like:

1: VijaySingh -3.840661; 90% interval: (-4.354664, -3.288281)
2: TigerWoods -3.737515; 90% interval: (-4.568503, -2.850548)
3: ErnieEls -3.430582; 90% interval: (-4.313688, -2.480963)
4: PhilMickelson -3.353783; 90% interval: (-3.961659, -2.734419)
5: StewartCink -3.044881; 90% interval: (-3.647329, -2.476737)
...
600: TommyAaron 4.883014; 90% interval: (2.666637, 7.017471)
601: DerekSanders 5.420959; 90% interval: (3.484806, 7.799434)
602: BobLohr 6.026131; 90% interval: (3.881693, 8.108147)
603: ArnoldPalmer 6.226227; 90% interval: (4.355256, 8.178954)
604: TimTims 7.978016; 90% interval: (5.905149, 10.156050)
"""

hypertournmean_candsd = 1  # variance
hypervar_candsd = 1  # variance
mean_candsd = 1  # variance
obsvar_candsd = 1  # variance

# tourns                    # list of tournament #s
# golfers                   # list of golfer names
# data                      # tuples of (name, score, tourn) from golfdataR.dat
# est_avg                   # estimated average score

data = []
for line in open('golfdataR.dat'):
    line_data = line.strip().split(' ')
    line_data[1] = float(line_data[1])
    data.append(line_data)

# data = [line.strip().split(' ') for line in open('golfdataR.dat')]

golfers = sorted(set([line[0] for line in data]))
scores = [float(line[1]) for line in data]
tourns = sorted(set([line[2] for line in data]), key=int)

est_avg = numpy.mean(scores)

#data = [item.strip() for item in (line.split(' ') for line in open('golfdataR.dat'))]


#data = (item.strip() for item in (line.split(' ')) for line in open('golfdataR.dat'))

hypertournmean = NormalNode(72.8, name='Tournament Hyper Mean', cand_var=hypertournmean_candsd, mean=72, var=2)
hypertournvar = InvGammaNode(3, name='Tournament Hyper Var', cand_var=hypervar_candsd, shape=18, scale=1 / .015)
tournmean = {}
for tourn in tourns:
    tournmean[tourn] = NormalNode(est_avg, name="Tournament {}".format(tourn), cand_var=mean_candsd,
                                  mean=hypertournmean, var=hypertournvar)

hypergolfervar = InvGammaNode(3.5, name='Golfer Hyper Var', cand_var=hypervar_candsd, shape=18, scale=1 / .015)
golfermean = {}
for golfer in golfers:
    golfermean[golfer] = NormalNode(est_avg, name=golfer, cand_var=mean_candsd, mean=0, var=hypergolfervar)

obsvar = InvGammaNode(3.1, name='Observation Var', cand_var=obsvar_candsd, shape=83, scale=1 / .0014)
for (name, score, tourn) in data:
    NormalNode(score, observed=True, mean=[tournmean[tourn], golfermean[name]], var=obsvar)

# sample from nodes
burn = 10000
nsamples = 100000

network = Network(
    [hypertournmean, hypertournvar, hypergolfervar, obsvar] + list(tournmean.values()) + list(golfermean.values()))
samples = network.collect_samples(burn, nsamples)

ability = []
for golfer in golfermean:
    golfermean_samples = samples.of_node(golfermean[golfer])[:]
    golfermean_samples.sort()
    median = golfermean_samples[nsamples // 2]
    low = golfermean_samples[int(.05 * nsamples)]
    high = golfermean_samples[int(.95 * nsamples)]
    ability.append((golfer, low, median, high))

ability = sorted(ability, key=itemgetter(2))  # sort by median score
i = 1
for golfer, low, median, high in ability:
    print("{}: {} {:.6f}; 90% interval: ({:.6f}, {:.6f})".format(i, golfer, median, low, high))
    i += 1