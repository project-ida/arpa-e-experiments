import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation
import pandas as pd

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


def plot_panels(combined_df, columns, start=None, stop=None, save_path=None, colors=None):
    # Example usage
    # fig, axes = plot_panels(combined_df, ['Thermocouple1Ch1', 'pressure_bar', 'loading'], 
    #                       start="2024-09-23 19:37:42", stop="2024-09-24 13:37:42", 
    #                       save_path="plot.png", colors=['blue', 'green', 'red'])
    # axes[0].set_ylabel("Custom Label") # Modify label after plotting, if needed
    
    # Convert start and stop times if provided, else use entire time range
    start_time = pd.to_datetime(start) if start else combined_df.index[0]
    end_time = pd.to_datetime(stop) if stop else combined_df.index[-1]

    # Set up the figure and axes based on the number of columns provided
    fig, axes = plt.subplots(nrows=len(columns), ncols=1, figsize=(8, 8), sharex=True)
    
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
        ax.plot(combined_df.loc[start_time:end_time, column], color=color)
        ax.set_ylabel(column.replace('_', ' ').title())
    
    # Set the x-axis label and rotate ticks for the last subplot
    axes[-1].set_xlabel('Time')
    axes[-1].tick_params(axis='x', rotation=30)

    # Safely get the descriptor from attrs, with a default if it doesn't exist
    descriptor = combined_df.attrs.get("descriptor", "Descriptor")
    fig.suptitle(f"{descriptor} {combined_df.index[0].date()}")

    # Adjust layout to prevent overlap
    fig.tight_layout()
    
    # Save the plot if a path is provided
    if save_path:
        fig.savefig(save_path, dpi=600)

    # Return the figure and axes for further customization
    return fig, axes


def plot_panels_with_scatter(combined_df, columns, scatter_x, scatter_y, start=None, stop=None, 
                      experiment_descriptor=None, colors=None, downsample=1, figsize=(12, 8), 
                             save_path=None, marker=None, animate=False, frames=10):
    """
    Plots time series data in a left column of subplots and a scatter plot in the right column.
    
    Parameters:
    - combined_df: DataFrame containing the data.
    - columns: List of column names for time series plots (left side).
    - scatter_x, scatter_y: Column names for x and y axes of the scatter plot (right side).
    - start, stop: Time range as strings (optional).
    - experiment_descriptor: String to add as a title (optional).
    - colors: List of colors for each time series subplot (optional).
    - downsample: Integer to downsample the data (every nth point).
    - figsize: Tuple for figure size.
    - save_path: Location of where the image should be saved 
    - marker: Time (ISO8601 string) where you want to highlight specific data points
    - animate: Boolean that tells matplotlib whether it should animate the markers
    - frames: Integer "n" that tells matplotlib to make a frame ever n points
    
    Returns:
    - fig, axes: Matplotlib figure and axes.
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

    # Safely get the descriptor from attrs, with a default if it doesn't exist
    descriptor = combined_df_downsampled.attrs.get("descriptor", "")
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