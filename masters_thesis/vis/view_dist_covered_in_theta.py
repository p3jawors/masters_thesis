from abr_analyze import DataHandler
from masters_thesis.utils import plotting
dat = DataHandler('llp_pd_d', 'data/databases')
data = dat.load(save_location='9999_linear_targets_faster', parameters=['state'])
plotting.plot_sliding_distance(
    positions=data['state'][:100000, :3],
    dt=0.01,
    theta=1
)

