import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation
import pandas as pd
import requests
from io import StringIO
from .auth import get_credentials

def load_data(url):
    """
    Loads a CSV file from a URL into a pandas DataFrame, handling password protection if required.

    This function first checks if the URL requires authentication by attempting an unauthenticated
    request. If successful (200 OK), it loads the data without credentials. If it receives a 401
    Unauthorized response, it uses stored credentials from the auth module to authenticate and
    retry the request. The CSV is parsed into a DataFrame with a time-based index.

    Parameters:
    -----------
    url : str
        The URL of the CSV file to be loaded. The file should be in CSV format with a 'time'
        column for the index, and may be password-protected.

    Returns:
    --------
    pandas.DataFrame or None
        A DataFrame with a time-based index ('time') if the data is loaded successfully.
        Returns None if authentication fails (e.g., 401 Unauthorized or missing credentials).

    Raises:
    -------
    RuntimeError
        If the HTTP request fails for reasons other than authentication (e.g., network issues,
        404 Not Found, or other server errors).

    Example:
    --------
    from libs.auth import authenticate
    from libs.helpers import load_data
    
    # For password-protected URL, authenticate first
    authenticate()  # Prompt for credentials if needed
    url = "https://example.com/protected_data.csv"
    df = load_data(url)
    if df is not None:
        print(df.head())

    # For unprotected URL, no authentication needed
    url = "https://example.com/public_data.csv"
    df = load_data(url)
    if df is not None:
        print(df.head())

    Notes:
    ------
    - Checks for password protection with an initial unauthenticated request.
    - If 401 is received, assumes authentication is required and uses credentials from authenticate().
    - The CSV file must have a 'time' column, which is parsed as a datetime and set as the index.
    - Uses ISO8601 format for parsing the 'time' column.
    - Credentials are optional; if not needed, the function skips authentication.
    - Useful for loading both public and protected experiment data with a consistent interface.
    """
    try:
        # Initial unauthenticated request to check if password is required
        response = requests.get(url)
        
        if response.status_code == 200:
            # No authentication required, process the response
            csv_content = response.text
            df = pd.read_csv(
                StringIO(csv_content),
                parse_dates=['time'],
                date_format="ISO8601",
                index_col='time'
            )
            print("Data file loaded successfully!")
            return df
        elif response.status_code == 401:
            # Authentication required, retrieve credentials and retry
            credentials = get_credentials()
            username = credentials['username']
            password = credentials['password']
            response = requests.get(url, auth=(username, password))
            if response.status_code == 200:
                csv_content = response.text
                df = pd.read_csv(
                    StringIO(csv_content),
                    parse_dates=['time'],
                    date_format="ISO8601",
                    index_col='time'
                )
                print("Data file loaded successfully with authentication!")
                return df
            else:
                print("Unauthorized: Please verify your username and password or re-run authenticate().")
                return None
        else:
            raise RuntimeError(f"Failed to retrieve the file. Status code: {response.status_code}")
    
    except ValueError as e:
        print("Authentication error:", str(e))
        return None
    except Exception as e:
        raise RuntimeError(f"Error loading data: {e}")

def print_info(dataframe):
    """
    Prints summary information about a DataFrame containing time-indexed measurements.

    This function outputs key information, including the start and end times of the data,
    total number of data points, average time between measurements, and a count of NaN
    values for each column.

    Parameters:
    -----------
    dataframe : pandas.DataFrame
        A DataFrame with a time-based index representing the measurement timestamps.
        The columns should represent various measured variables.

    Outputs:
    --------
    - Start time of measurements.
    - End time of measurements.
    - Total number of data points.
    - Average time interval (in seconds) between data points.
    - Count of NaN values per column.

    Example:
    --------
    print_info(dataframe)
    
    Notes:
    ------
    - Assumes that `dataframe` has a time-based index in chronological order.
    - Calculates the time interval by dividing the total time span by the number of points.
    - Useful for quickly inspecting data completeness and identifying potential data gaps.
    """
    
    # When does data collection begin and end
    print(f"Measurements start at: {dataframe.index[0]}")
    print(f"Measurements end at: {dataframe.index[-1]}")
    print("---------")
    
    # How many data points do we have
    raw_total_points = len(dataframe)
    print(f"Total number of measurements: {raw_total_points}")

    # What's the average time in seconds time between data points
    time_between_points = ((dataframe.index[-1] - dataframe.index[0]) / raw_total_points).total_seconds()
    print(f"Time between measurements: {time_between_points} s")
    print("---------")

    # Count problem values like NaNs
   
    print("Total number of NaNs")
    print(dataframe.isna().sum())
    
    

def process_data(dataframes, meta_data=None):
    """
    Processes multiple time-indexed DataFrames by aligning them to a common time index,
    handling duplicate timestamps, and applying metadata to the resulting DataFrame.

    Parameters:
    ----------
    dataframes : list of pd.DataFrame
        A list of DataFrames to be processed. Each DataFrame is expected to have a time-based
        index, which will be rounded to the nearest second and averaged for any duplicate
        timestamps after rounding.

    meta_data : dict, optional
        A dictionary containing metadata to attach to the final DataFrame. Each key-value pair
        in `meta_data` will be added to the `attrs` attribute of the returned DataFrame, allowing
        easy access to metadata (e.g., source information, experiment details).
        
        If no meta_data is provided (default is None), a reminder message will print, suggesting
        how users can add descriptive metadata.

    Returns:
    -------
    pd.DataFrame
        A single DataFrame with a common time index across all input DataFrames. The data
        from each input DataFrame is concatenated along columns, with missing values interpolated
        linearly. The `attrs` attribute of the returned DataFrame contains the metadata provided
        by `meta_data` if supplied.

    Notes:
    -----
    - The function aligns the DataFrames to a shared time range based on the maximum starting
      timestamp and minimum ending timestamp across all DataFrames.
    - Duplicate timestamps within each DataFrame are averaged after rounding to the nearest second.
    - Any remaining NaN values at the edges of the combined DataFrame are dropped after interpolation.
    
    Example:
    -------
    >>> temperature_df = pd.DataFrame({...}, index=pd.to_datetime([...]))
    >>> pressure_df = pd.DataFrame({...}, index=pd.to_datetime([...]))
    >>> meta_data = {"descriptor": "Etched titanium foil"}
    >>> combined_df = process_data([temperature_df, pressure_df], meta_data)
    >>> print(combined_df.attrs["descriptor"])
    'Etched titanium foil'
    """

    # Step 1: Round the time index and average duplicates within each DataFrame
    processed_dfs = []
    for df in dataframes:
        df = df.copy()  # Avoid modifying the original DataFrame
        df.index = df.index.floor('s') # Round the time to the nearest second
        df = df.groupby(df.index).mean() # Average any values at the same time after rounding
        processed_dfs.append(df)
    
    # Step 2: Determine the global overlapping time range
    start_time = max(df.index.min() for df in processed_dfs)
    end_time = min(df.index.max() for df in processed_dfs)
    
    # Step 3: Limit each DataFrame to the overlapping time range
    processed_dfs = [
        df[(df.index >= start_time) & (df.index <= end_time)]
        for df in processed_dfs
    ]
    
    # Step 4: Concatenate along columns to merge by time index
    combined_df = pd.concat(processed_dfs, axis=1)
    
    # Step 5: Interpolate missing values and drop any remaining NaNs at the time range edges
    combined_df = combined_df.interpolate(method='linear').dropna()

    # Step 6: Attach metadata to the DataFrame if provided
    if meta_data is not None:
        combined_df.attrs.update(meta_data)
    else:
        # Print a reminder message for the user
        print("No metadata provided. Consider adding metadata for context. Example:")
        print('meta_data = {"descriptor": "Etched titanium foil"}  # Useful for plot titles')
    
    return combined_df


def plot_panels(combined_df, columns, start=None, stop=None, save_path=None, colors=None, downsample=1, figsize=(8, 8), marker=None):
    """
    Plots multiple time series columns from a DataFrame as individual subplots.

    Parameters:
    -----------
    combined_df : pandas.DataFrame
        A DataFrame with a time-based index and columns to be plotted. 
        The DataFrame should have attributes, including a `descriptor` (optional),
        which, if provided and non-empty, will be used as the figure's title.

    columns : list of str
        A list of column names from `combined_df` to be plotted. Each column 
        will appear on a separate subplot.

    start : str or pandas.Timestamp, optional
        The start time for the data to be plotted. If None, the plotting 
        will begin from the start of the DataFrame's index.

    stop : str or pandas.Timestamp, optional
        The end time for the data to be plotted. If None, the plotting 
        will continue until the end of the DataFrame's index.

    save_path : str, optional
        File path to save the generated plot as an image. If None, the plot 
        will not be saved.

    colors : list of str, optional
        A list of colors for each subplot. If fewer colors than columns are provided,
        the remaining subplots will use the color blue by default. If None, 
        all subplots will use the color blue.

    downsample : int, optional
        Factor by which to downsample the data, plotting every `downsample`-th point. Default is 1 (no downsampling).

    figsize : tuple, optional
        Figure size for the plot, default is (12, 8).
        
    marker : str (ISO8601) or pandas.Timestamp, optional
        Specific timestamp to mark on the plot. A vertical line will appear in the 
        time series plots

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object for the generated plot.

    axes : list of matplotlib.axes.Axes
        A list of axes objects, each corresponding to a subplot for each 
        specified column in `columns`. 

    Example:
    --------
    fig, axes = plot_panels(combined_df, ['Thermocouple1Ch1', 'pressure_bar', 'loading'], 
                            start="2024-09-23 19:37:42", stop="2024-09-24 13:37:42", 
                            save_path="plot.png", colors=['blue', 'green', 'red'], marker="2024-09-23 20:00")
    axes[0].set_ylabel("Custom Label")  # Modify labels as needed after plotting.

    Notes:
    ------
    - The x-axis is shared across subplots, and labels on the last subplot 
      are rotated for clarity.
    - The `descriptor` attribute in `combined_df` is used as the figure's 
      title if it is non-empty.
    """

    # Downsample the data
    combined_df_downsampled = combined_df.iloc[::downsample]
    
    # Convert start and stop times if provided, else use entire time range
    start_time = pd.to_datetime(start) if start else combined_df_downsampled.index[0]
    end_time = pd.to_datetime(stop) if stop else combined_df_downsampled.index[-1]

    # Restrict data to between start_time and end_time
    combined_df_downsampled = combined_df_downsampled[start_time:end_time]

    # Set up the figure and axes based on the number of columns provided
    fig, axes = plt.subplots(nrows=len(columns), ncols=1, figsize=figsize, sharex=True)
    
    # If there's only one column, axes won't be an array, so we ensure it's iterable
    if len(columns) == 1:
        axes = [axes]

    # Set colors default to blue if not provided or incomplete
    if colors is None:
        colors = ['blue'] * len(columns)
    else:
        colors = colors + ['blue'] * (len(columns) - len(colors))  # Extend with blue if fewer colors are given

    # Plot each specified column on its own subplot
    for ax, column, color in zip(axes, columns, colors):
        ax.plot(combined_df_downsampled[column], color=color)
        ax.set_ylabel(column.replace('_', ' ').title())
    
    # Set the x-axis label and rotate ticks for the last subplot
    axes[-1].set_xlabel('Time')
    axes[-1].tick_params(axis='x', rotation=30)

    # Plots a maker as vertical lines on the panels or a point on the scatter
    if marker:
        marker_datetime = pd.to_datetime(marker)
        closest_index = combined_df_downsampled.index.asof(marker_datetime)

        vertical_lines = [ax.axvline(x=closest_index, color='black', linestyle='--') for ax in axes]

    # Safely get the descriptor from attrs and add title only if descriptor is not empty
    descriptor = combined_df_downsampled.attrs.get("descriptor", "")
    if descriptor:  # Check if descriptor is not empty
        fig.suptitle(f"{descriptor} {combined_df_downsampled.index[0].date()}")

    # Adjust layout to prevent overlap
    fig.tight_layout()
    
    # Save the plot if a path is provided
    if save_path:
        fig.savefig(save_path, dpi=600)

    # Return the figure and axes for further customization
    return fig, axes


def plot_panels_with_scatter(combined_df, columns, scatter_x, scatter_y, start=None, stop=None, 
                             colors=None, downsample=1, figsize=(12, 8), 
                             save_path=None, marker=None, animate=False, frames=10):
    """
    Plots a series of time series data as individual subplots in a left column and a scatter plot 
    of two variables in a right column. Optionally, a specific point can be marked on the plot 
    or an animation can be created to highlight points over time.

    Parameters:
    -----------
    combined_df : pandas.DataFrame
        DataFrame containing time-indexed data for both time series and scatter plot columns.
        Attributes may include a 'descriptor', which will be used as a title if not empty.

    columns : list of str
        List of column names from `combined_df` to be plotted as time series in the left column.

    scatter_x : str
        Column name for the x-axis variable in the scatter plot.

    scatter_y : str
        Column name for the y-axis variable in the scatter plot.

    start : str or pandas.Timestamp, optional
        Start time for the data to be plotted. If None, the plotting begins from the 
        start of the DataFrame's index.

    stop : str or pandas.Timestamp, optional
        End time for the data to be plotted. If None, the plotting continues to the 
        end of the DataFrame's index.

    colors : list of str, optional
        List of colors for each time series subplot. If fewer colors than columns are provided, 
        remaining subplots use blue by default. If None, all subplots use blue.

    downsample : int, optional
        Factor by which to downsample the data, plotting every `downsample`-th point. Default is 1 (no downsampling).

    figsize : tuple, optional
        Figure size for the plot, default is (12, 8).

    save_path : str, optional
        File path to save the plot as an image. If None, the plot will not be saved.

    marker : str (ISO8601) or pandas.Timestamp, optional
        Specific timestamp to mark on the plot. A vertical line will appear in the time series 
        plots, and a point will be highlighted in the scatter plot.

    animate : bool, optional
        Whether to animate the marker moving across time series and scatter plot data. Default is False.

    frames : int, optional
        Frame interval for the animation, setting how many points to skip between frames.
        Default is 10.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object for the generated plot.

    time_axes : list of matplotlib.axes.Axes
        List of axes for the time series subplots.

    ax_scatter : matplotlib.axes.Axes
        Axes object for the scatter plot.

    Example:
    --------
    fig, time_axes, ax_scatter = plot_panels_with_scatter(
        combined_df, ['Temp', 'Pressure'], scatter_x='humidity', scatter_y='vibration',
        start="2024-09-23 19:37:42", stop="2024-09-24 13:37:42", colors=['blue', 'green'], 
        downsample=10, save_path="plot.png", marker="2024-09-24 00:00:00",
        animate=True, frames=10
    )

    Notes:
    ------
    - The time series plots are displayed in a column on the left, while a scatter plot of two 
      specified variables is displayed in a column on the right.
    - The animation (if enabled) highlights points in both time series and scatter plot as they 
      progress over time. Requires 'ffmpeg' for saving the animation.
    - Use the DataFrame's `descriptor` attribute for a title, which will only display if not empty.
    """

    # Downsample the data
    combined_df_downsampled = combined_df.iloc[::downsample]

    # Convert start and stop times if provided, else use entire time range
    start_time = pd.to_datetime(start) if start else combined_df_downsampled.index[0]
    end_time = pd.to_datetime(stop) if stop else combined_df_downsampled.index[-1]

    # Restrict data to between start_time and end_time
    combined_df_downsampled = combined_df_downsampled[start_time:end_time]

    # Set up the figure and GridSpec layout
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(len(columns), 2, width_ratios=[2, 1])  # Define a grid with 2:1 width ratio for time series and scatter plot

    # Set colors to blue if not provided or incomplete
    if colors is None:
        colors = ['blue'] * len(columns)
    else:
        colors = colors + ['blue'] * (len(columns) - len(colors))

    # Create subplots for time series in the left column
    time_axes = []
    for i, (column, color) in enumerate(zip(columns, colors)):
        ax = fig.add_subplot(gs[i, 0], sharex=time_axes[0] if i > 0 else None)
        ax.plot(combined_df_downsampled[column], color=color)
        ax.set_ylabel(column.replace('_', ' ').title())
        
        # Hide x-tick labels for all but the last subplot
        if i < len(columns) - 1:
            ax.tick_params(labelbottom=False)
            
        time_axes.append(ax)

    # Set the x-axis label and rotate ticks for the last subplot
    time_axes[-1].set_xlabel('Time')
    time_axes[-1].tick_params(axis='x', rotation=30)

    # Create the scatter plot in the right column
    ax_scatter = fig.add_subplot(gs[:, 1])  # Use all rows in the second column
    ax_scatter.scatter(combined_df_downsampled[scatter_x],
                       combined_df_downsampled[scatter_y],
                       color='orange', alpha=0.7)
    ax_scatter.set_xlabel(scatter_x.replace('_', ' ').title())
    ax_scatter.set_ylabel(scatter_y.replace('_', ' ').title())
    ax_scatter.yaxis.tick_right()
    ax_scatter.yaxis.set_label_position("right")

    # Plots a maker as vertical lines on the panels or a point on the scatter
    if marker:
        marker_datetime = pd.to_datetime(marker)
        closest_index = combined_df_downsampled.index.asof(marker_datetime)

        vertical_lines = [ax.axvline(x=closest_index, color='black', linestyle='--') for ax in time_axes]
        scatter_point = ax_scatter.scatter([combined_df_downsampled[scatter_x][closest_index]], 
                                           [combined_df_downsampled[scatter_y][closest_index]], color='blue', s=50)

    # Safely get the descriptor from attrs and add title only if descriptor is not empty
    descriptor = combined_df.attrs.get("descriptor", "")
    if descriptor:  # Check if descriptor is not empty
        fig.suptitle(f"{descriptor} {combined_df_downsampled.index[0].date()}")

    # Adjust layout to prevent overlap
    fig.tight_layout()

    # Animate the makers
    if animate:
        # Initialize moving elements (vertical lines for each panel and a moving point in scatter plot)
        vertical_lines = [ax.axvline(x=combined_df_downsampled.index[0], color='black', linestyle='--') for ax in time_axes]
        moving_point = ax_scatter.scatter([], [], color='blue', s=50)

        # Define the update function
        def update(frame):
            current_time = combined_df_downsampled.index[frame]
        
            # Update positions of the vertical lines
            for line in vertical_lines:
                line.set_xdata([current_time])
        
            # Update the scatter plot indicator position
            scatter_x_value = combined_df_downsampled[scatter_x].iloc[frame]
            scatter_y_value = combined_df_downsampled[scatter_y].iloc[frame]
            moving_point.set_offsets([[scatter_x_value, scatter_y_value]])
        
            return *vertical_lines, moving_point
            
        # Create the animation
        ani = animation.FuncAnimation(fig, update, frames=range(0, len(combined_df_downsampled), frames), interval=200, blit=True)

        # Close the figure to suppress the static image
        plt.close(fig)

        # Safely get the descriptor from attrs, with a default if it doesn't exist
        descriptor = combined_df_downsampled.attrs.get("descriptor", "")

        # Animation needs to be saved in order for it to be viewed in another cell via Video()
        ani.save(f"media/{descriptor} {combined_df_downsampled.index[0].date()}.mp4", writer='ffmpeg', fps=30)

        # Tells the user where the file has been saved so it's easy to display using Video()
        print(f"Animation saved at 'media/{descriptor} {combined_df_downsampled.index[0].date()}.mp4'")

    # Save the plot if a path is provided
    if save_path:
        fig.savefig(f"media/{save_path}", dpi=600)

    # Return the figure and axes for further customization or saving
    return fig, time_axes, ax_scatter