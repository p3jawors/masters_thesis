from abr_analyze import DataHandler
from masters_thesis.utils import plotting

dat = DataHandler('llp_pd', 'data/databases')
save_location='100_linear_targets'
data = dat.load(
    save_location=save_location,
    parameters=dat.get_keys(save_location)
)
folder = 'data/presentation_figures/'

plotting.plot_2d(
    data['time'],
    data['state'],
    labels=['x', 'y', 'z', 'dx', 'dy', 'dz', 'a', 'b', 'g', 'da', 'db', 'dg'],
    title='World State',
    n_rows=4,
    save_name=f'{folder}world_state_over_time.png'
)

plotting.plot_2d(
    data['time'],
    data['mean_shift_abs_max_scale_state'],
    labels=['x', 'y', 'z', 'dx', 'dy', 'dz', 'a', 'b', 'g', 'da', 'db', 'dg'],
    title='World State: Mean Shifted and Normalized',
    save_name=f"{folder}world_state-shifted_norm_over_time.png"
)


plotting.plot_2d(
    data['time'],
    data['ego_error'],
    labels=['x', 'y', 'z', 'dx', 'dy', 'dz', 'a', 'b', 'g', 'da', 'db', 'dg'],
    title='Local State',
    save_name=f'{folder}local_state_over_time.png',
    n_rows=4
)

plotting.plot_2d(
    data['time'],
    data['mean_shifted_normalized_ego_error'],
    labels=['x', 'y', 'z', 'dx', 'dy', 'dz', 'a', 'b', 'g', 'da', 'db', 'dg'],
    title='Local State: Mean Shifted and Normalized',
    save_name=f"{folder}local_state-shifted_norm_over_time.png",
)


plotting.plot_2d(
    data['time'],
    data['ctrl'],
    labels=['u_front_right', 'u_rear_left', 'u_front_left', 'u_rear_right'],
    title='Control Signal',
    n_rows=4,
    save_name=f'{folder}control-over_time.png'
)

plotting.plot_2d(
    data['time'],
    data['clean_u'],
    labels=['u_front_right', 'u_rear_left', 'u_front_left', 'u_rear_right'],
    title='Control Signal: Gravity Removed, Clipped, Normalized',
    n_rows=4,
    save_name=f'{folder}control-gravity_removed_clipped_norm_over_time.png'
)
