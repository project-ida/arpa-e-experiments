from .io import (
    get_roi_name_from_api_url,
    load_roi_api,
    infer_dataset_base_from_api,
    add_json_urls,
    fetch_json_items,
    build_spectrum_index,
    attach_spectra,
    get_selection_grid_url,
    load_all_cells_from_selection_grid,
)

from .spectra import (
    stack_spectra,
    stack_spectra_trim,
    band_sum,
    summarize_band_values,
    resolve_band_to_channels,
    band_label_text,
    print_cli_suggestions,
)

from .calibration import (
    load_config_txt,
    get_energy_cal_from_dataset,
    make_energy_axis,
    make_energy_axis_from_length,
    channel_to_keV,
    keV_to_channel,
    maybe_get_calibration,
)

from .peaks import (
    baseline_als,
    preprocess,
    estimate_noise,
    detect_peaks,
    line_library,
    identify_elements,
)

from .plotting import (
    get_plot_axis,
    get_band_span,
    add_energy_top_axis,
    plot_cumulative,
    plot_overlay,
    plot_with_peaks,
    plot_identified_elements_confident,
)

from .overlays import (
    get_api_auth,
    create_overlay,
    delete_overlay,
)
