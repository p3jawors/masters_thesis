from abr_analyze import DataHandler
from masters_thesis.utils import plotting
import sys

if len(sys.argv) > 1:
    if len(sys.argv) > 2:
        start = int(sys.argv[1])
        end = int(sys.argv[2])
    else:
        start = 0
        end = int(sys.argv[1])
else:
    start = 0
    end = -1
dat = DataHandler('llp_pd_d', 'data/databases')
save_location='9999_linear_targets_faster'
# dat = DataHandler('llp_pd', 'data/databases')
# save_location='100_linear_targets'
data = dat.load(
    save_location=save_location,
    parameters=dat.get_keys(save_location)
)
folder = 'data/presentation_figures/'

plotting.plot_2d(
        data['time'][start:end],
        data['state'][start:end, :],
    labels=['x', 'y', 'z', 'dx', 'dy', 'dz', 'a', 'b', 'g', 'da', 'db', 'dg'],
    title='World State',
    n_rows=4,
    save_name=f'{folder}world_state_over_time.png'
)

plotting.plot_2d(
    data['time'][start:end],
    data['mean_shift_abs_max_scale_state'][start:end],
    labels=['x', 'y', 'z', 'dx', 'dy', 'dz', 'a', 'b', 'g', 'da', 'db', 'dg'],
    title='World State: Mean Shifted and Normalized',
    save_name=f"{folder}world_state-shifted_norm_over_time.png"
)


plotting.plot_2d(
        data['time'][start:end],
        data['ego_error'][start:end, :],
    labels=['x', 'y', 'z', 'dx', 'dy', 'dz', 'a', 'b', 'g', 'da', 'db', 'dg'],
    title='Local State',
    save_name=f'{folder}local_state_over_time.png',
    n_rows=4
)

plotting.plot_2d(
        data['time'][start:end],
        data['mean_shifted_normalized_ego_error'][start:end, :],
    labels=['x', 'y', 'z', 'dx', 'dy', 'dz', 'a', 'b', 'g', 'da', 'db', 'dg'],
    title='Local State: Mean Shifted and Normalized',
    save_name=f"{folder}local_state-shifted_norm_over_time.png",
)


plotting.plot_2d(
        data['time'][start:end],
        data['ctrl'][start:end, :],
    labels=['u_front_right', 'u_rear_left', 'u_front_left', 'u_rear_right'],
    title='Control Signal',
    n_rows=4,
    save_name=f'{folder}control-over_time.png'
)

plotting.plot_2d(
        data['time'][start:end],
        data['clean_u'][start:end, :],
    labels=['u_front_right', 'u_rear_left', 'u_front_left', 'u_rear_right'],
    title='Control Signal: Gravity Removed, Clipped, Normalized',
    n_rows=4,
    save_name=f'{folder}control-gravity_removed_clipped_norm_over_time.png'
)
