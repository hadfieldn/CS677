hypertournmean = NormalNode(72.8, name='Tournament Hyper Mean', \
        candsd=hypertournmean_candsd, mean=72, var=2)
hypertournvar = InvGammaNode(3, name='Tournament HyperVar', \
        candsd=hypervar_candsd, shape=18, scale=1/.015)
tournmean = {}
for tourn in tourns:
    tournmean[tourn] = NormalNode(est_avg,
                                  name='Tournament %s'%tourn, \
                                  candsd=mean_candsd, mean=hypertournmean,
                                  var=hypertournvar)
hypergolfervar = InvGammaNode(3.5, name='Golfer Hyper Var', \
                              candsd=hypervar_candsd, shape=18, scale=1/.015)
golfermean = {}
for golfer in golfers:
    golfermean[golfer] = NormalNode(est_avg, name=golfer, \
                                    candsd=mean_candsd, mean=0, var=hypergolfervar)
obsvar = InvGammaNode(3.1, name='Observation Var', \
                      candsd=obsvar_candsd, shape=83, scale=1/.0014)
for (name, score, tourn) in data:
    NormalNode(score, observed=True, \
               mean=tournmean[tourn]+golfermean[name],
               var=obsvar)