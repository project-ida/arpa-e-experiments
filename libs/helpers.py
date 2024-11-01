import matplotlib.pyplot as plt
import pandas as pd

def process_data(*dataframes):
    # Example usage
    # combined_df = process_data(temperature_df, pressure_df)
    
    # Step 1: Round the time index and average duplicates within each DataFrame
    processed_dfs = []
    for df in dataframes:
        df = df.copy()  # Avoid modifying the original DataFrame
        df.index = df.index.floor('s') # Round the time to the nearest second
        df = df.groupby(df.index).mean() # Average any temperature values that occur at the same time now that we've rounded the time stamp to the nearest second
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
    
    # Step 5: Interpolate missing values and drop any remaining NaNs that might occur at the start of end of the time range
    combined_df = combined_df.interpolate(method='linear').dropna()
    
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