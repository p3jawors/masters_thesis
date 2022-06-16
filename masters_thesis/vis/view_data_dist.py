"""
Plots the data distribution of the raw and cleaned up state and control signals
"""
import numpy as np
from masters_thesis.utils import plotting
from abr_analyze import DataHandler

# dat = DataHandler('llp_pd_d', 'data/databases')
# save_location='9999_linear_targets_faster'

dat = DataHandler('llp_pd', 'data/databases')
save_location='100_linear_targets'
data = dat.load(
    save_location=save_location,
    parameters=dat.get_keys(save_location)
)
folder = 'data/presentation_figures/'

# plotting.plot_data_distribution(
#     data['state'],
#     dim_labels=['x', 'y', 'z', 'dx', 'dy', 'dz', 'a', 'b', 'g', 'da', 'db', 'dg'],
#     title='World State',
#     save_name=f"{folder}world_state-distribution.png"
#
# )
# plotting.plot_data_distribution(
#     data['mean_shift_abs_max_scale_state'],
#     dim_labels=['x', 'y', 'z', 'dx', 'dy', 'dz', 'a', 'b', 'g', 'da', 'db', 'dg'],
#     title='World State: Mean Shifted and Normalized',
#     save_name=f"{folder}world_state-shifted_norm_distribution.png"
# )
#
# plotting.plot_data_distribution(
#     data['ego_error'],
#     dim_labels=['x', 'y', 'z', 'dx', 'dy', 'dz', 'a', 'b', 'g', 'da', 'db', 'dg'],
#     title='Local State',
#     save_name=f"{folder}local_state-distribution.png"
# )
#
# plotting.plot_data_distribution(
#     data['mean_shifted_normalized_ego_error'],
#     dim_labels=['x', 'y', 'z', 'dx', 'dy', 'dz', 'a', 'b', 'g', 'da', 'db', 'dg'],
#     title='Local State: Mean Shifted and Normalized',
#     save_name=f"{folder}local_state-shifted_norm_distribution.png",
# )
#
# plotting.plot_data_distribution(
#     data['ctrl'],
#     dim_labels=['u_front_right', 'u_rear_left', 'u_front_left', 'u_rear_right'],
#     title='Control Signal',
#     save_name=f"{folder}control-distribution.png"
# )
#
plotting.plot_data_distribution(
    data['clean_u'],
    dim_labels=['u_front_right', 'u_rear_left', 'u_front_left', 'u_rear_right'],
    title='Control Signal: Gravity Removed, Clipped, Normalized',
    save_name=f"{folder}control-gravity_removed_clipped_norm_distribution.png"
)

# print('dt: ', np.mean(np.diff(data['time'])))
# print('shape: ', data['clean_u'].shape)
# print('shape: ', np.diff(data['clean_u'], axis=0).shape)
plotting.plot_data_distribution(
    # np.diff(data['clean_u'])/np.mean(np.diff(data['time'])),
    np.diff(data['clean_u'], axis=0),
    dim_labels=['u_front_right', 'u_rear_left', 'u_front_left', 'u_rear_right'],
    title='Rate of Change of Control Signal: Gravity Removed, Clipped, Normalized',
    save_name=f"{folder}rate_of_change_control-gravity_removed_clipped_norm_distribution.png",
    bins=100
)

