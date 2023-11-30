# Import Statements
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import scipy.stats as stats
import math

"""
This program performs an analysis of anthropometric data, visualizing the distributions 
of weight-for-age (wfa), height-for-age (hfa), and BMI-for-age (bmifa) z-scores across 
a population. It includes the calculation of mean z-scores and confidence 
intervals, as well as the prevalence of different malnutrition indicators. Results are 
visualized using KDE plots with shaded areas representing malnourished populations 
according to defined thresholds.

Usage:
- The companion program, AnthroSnake, should be used to compute z-scores from raw 
  anthropometric data. The output of AnthroSnake should be the input of this program.
- Execute the script to perform the analysis and generate plots. Modify the 
  DATA_FILE_NAME constant at the top of the script to change the input data source.

Requirements:
- pandas (as pd): For data manipulation and analysis.
- numpy (as np): For numerical computations.
- matplotlib.pyplot (as plt), matplotlib.gridspec, matplotlib.patches: For creating and customizing 
  static plots in Python.
- seaborn (as sns): For making attractive and informative statistical graphics.
- scipy.stats (as stats): For performing statistical computations.
- math: For basic mathematical functions and constants.
"""

# Constants
COLORS = {'yellow': '#FFDFBA', 'red': '#FFB3BA'}
TEXT_X_COORDINATE = (-0.05, 0.23, 0.6)
DATA_FILE_NAME = "output_with_z_scores.csv"

# Function Definitions
def import_and_prepare_data(data_file_name):
    """
    Imports data from a CSV file and prepares it for analysis by creating pandas Series for various anthropometric indicators.

    Args:
    - data_file_name (str): The name of the CSV file containing the data.

    Returns:
    - Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]: A tuple containing the Series for weight-for-age (wfa), 
      height-for-age (hfa), BMI-for-age (bmifa), BMI-for-age for subjects over 60 months (over_60_mo_bmifa), and age.
    """

    data_with_z_scores = pd.read_csv(data_file_name, index_col=0)

    # Isolate data on kids over 60 months old for overweight/obese prevalence
    data_over_60_months = data_with_z_scores[data_with_z_scores.age_months > 60].reset_index(drop=True)

    # Create a series for the z-scores of each indicator
    wfa_series = data_with_z_scores["wfa_z_score"]
    wfa_series = wfa_series[wfa_series.notna()].reset_index(drop=True) #Remove subjects over 10 years old
    hfa_series = data_with_z_scores["hfa_z_score"]
    bmifa_series = data_with_z_scores["bmifa_z_score"]
    over_60_mo_bmifa_series = data_over_60_months["bmifa_z_score"]

    # Create a series for participant ages
    age_series = data_with_z_scores["age_months"]

    return wfa_series, hfa_series, bmifa_series, over_60_mo_bmifa_series, age_series

def create_figures():
    """
    Creates two matplotlib figures with specific GridSpecs for plotting anthropometric data.

    Returns:
    - Tuple[plt.Figure, gridspec.GridSpec, plt.Figure, gridspec.GridSpec]: Two figures and their corresponding
      GridSpec objects. The first figure (fig1) and its GridSpec are configured for three main plots with additional 
      space for text annotations. The second figure (fig2) and its GridSpec (gs2) are set up for two main plots with 
      additional space for text annotations.

    """

    fig1 = plt.figure(figsize=(8, 15))
    gs1 = gridspec.GridSpec(6, 1, height_ratios=[1, 0.3, 1, 0.3, 1, 0.3])
    blank_ax = fig1.add_subplot(gs1[5]) # Create space for text annotations below lowest set of axes
    blank_ax.axis('off')

    fig2 = plt.figure(figsize=(8, 9))
    gs2 = gridspec.GridSpec(3, 1, height_ratios=[1, 0.3, 1])

    return fig1, gs1, fig2, gs2

def plot_z_distribution_kde(data, title, x_label, fig, position):
    """
    Plots a KDE distribution.
    
    Args:
    - data (Series): The data to plot.
    - title (str): The title of the plot.
    - x_label (str): The label for the x-axis.
    - fig (Figure): The figure to plot on.
    - position (int): The subplot position.

    Returns:
    - ax (Axes): The Axes object of the plot.
    """

    ax = fig.add_subplot(position)
    sns.kdeplot(data, ax=ax, fill=True)
    ax.set_title(title)
    ax.set_xticks(np.linspace(-5, 5, num=11))
    ax.set_xlim(left=-5, right=5)
    ax.set_xlabel(x_label)

    return ax

def shade_prevalence_on_distribution(data, ax, threshold, color, above=False):
    """
    Shades a portion of a given distribution based on an x-value threshold.

    Args:
    - data (Series): The data to plot.
    - ax (Axes): The Axes object to plot on.
    - threshold (float): The threshold x-value for shading.
    - color (str): The color of the shaded area.
    - above (bool): Whether to shade above the threshold. Defaults to False. 
    """

    poly_collections = ax.collections
    color_to_match = poly_collections[0].get_facecolor()[0]
    kde = sns.kdeplot(data, ax=ax, color=color_to_match)
    x_values = kde.get_lines()[0].get_xdata()
    y_values = kde.get_lines()[0].get_ydata()
    if not above:
        mask = x_values <= threshold
    else:
        mask = x_values >= threshold
    x_fill = x_values[mask]
    y_fill = y_values[mask]
    kde.fill_between(x_fill, y_fill, color=color, alpha=1)

def plot_data(wfa_series, hfa_series, bmifa_series, over_60_mo_bmifa_series, age_series, fig1, gs1, fig2, gs2):
    """
    Plots anthropometric data on the provided figures using the specified GridSpecs.

    Args:
    - wfa_series, hfa_series, bmifa_series, over_60_mo_bmifa_series, age_series (pd.Series): Series containing the data to be plotted.
    - fig1, fig2 (plt.Figure): The figures on which to plot the data.
    - gs1, gs2 (gridspec.GridSpec): GridSpecs defining the layout of the plots within the figures.

    Returns:
    - Tuple[plt.Axes]: Axes objects for each of the main plots created.
    """

    wfa_ax = plot_z_distribution_kde(wfa_series, "Distribution of Weight-For-Age Z-Scores, Ages 0-10", "Z-score", fig1, gs1[0])
    hfa_ax = plot_z_distribution_kde(hfa_series, "Distribution of Height-For-Age Z-Scores", "Z-score", fig1, gs1[2])
    bmifa_ax = plot_z_distribution_kde(bmifa_series, "Distribution of BMI-For-Age Z-Scores", "Z-score", fig1, gs1[4])

    over_60_mo_bmifa_ax = plot_z_distribution_kde(over_60_mo_bmifa_series, "Distribution of BMI-For-Age Z-Scores, Ages 5-19", "Z-score", fig2, gs2[0])

    age_ax = fig2.add_subplot(gs2[2])
    sns.kdeplot(age_series, ax=age_ax, fill=True)
    age_ax.set_title("Distribution of Participant Ages")
    age_ax.set_xlabel("Age (Months)")

    # Shade prevalences of malnourishment:
    shade_prevalence_on_distribution(wfa_series, wfa_ax, threshold=-2, color=COLORS['yellow']) # Underweight
    shade_prevalence_on_distribution(wfa_series, wfa_ax, threshold=-3, color=COLORS['red']) # Severely underweight
    shade_prevalence_on_distribution(hfa_series, hfa_ax, threshold=-2, color=COLORS['yellow']) # Stunting
    shade_prevalence_on_distribution(hfa_series, hfa_ax, threshold=-3, color=COLORS['red']) # Severely stunting
    shade_prevalence_on_distribution(bmifa_series, bmifa_ax, threshold=-2, color=COLORS['yellow']) # Thin
    shade_prevalence_on_distribution(over_60_mo_bmifa_series, over_60_mo_bmifa_ax, threshold=1, above=True, color=COLORS['yellow']) # Overweight
    shade_prevalence_on_distribution(over_60_mo_bmifa_series, over_60_mo_bmifa_ax, threshold=2, above=True, color=COLORS['red']) # Obese

    return wfa_ax, hfa_ax, bmifa_ax, over_60_mo_bmifa_ax, age_ax

def compute_mean_with_CI(data, confidence):
    """
    Computes the mean and confidence interval for a dataset.
    
    Args:
    - data (Series): The data to compute the mean and CI for.
    - confidence (float): The confidence level for the CI. Value between 0 and 1.

    Returns:
    - tuple: Mean and confidence interval.
    """

    n = len(data)
    mean = data.mean()
    mean_CI = stats.t.interval(confidence=confidence,
                               df=n-1, loc=mean,
                               scale=(data.std()/math.sqrt(n)))
    return mean, mean_CI

def compute_prevalence_rate_with_CI(data, threshold, confidence, above=False):
    """
    Computes the prevalence rate and confidence interval for a dataset.

    Args:
    - data (Series): The data to compute the prevalence for.
    - threshold (float): The threshold for determining prevalence
    - confidence (float): The confidence level for the CI. Value between 0 and 1.
    - above (bool): Whether to consider values above the threshold. Defaults to False.

    Returns:
    - tuple: Prevalence rate and confidence interval.
    """

    n = len(data)
    if not above:
        num_malnourished = (data < threshold).sum()
    else:
        num_malnourished = (data > threshold).sum()
    p_malnourished = (num_malnourished/n)
    p_malnourished_CI = stats.norm.interval(confidence=confidence,
                                            loc=p_malnourished,
                                            scale=math.sqrt((p_malnourished*(1-p_malnourished))/n))
    return p_malnourished, p_malnourished_CI

def compute_prevalences(wfa_series, hfa_series, bmifa_series, over_60_mo_bmifa_series):
    """
    Computes prevalences of various malnourishment indicators.

    Args:
    - wfa_series, hfa_series, bmifa_series, over_60_mo_bmifa_series (pd.Series): Series containing the anthropometric data.

    Returns:
    - dict: A dictionary containing the prevalence data and confidence intervals for each malnourishment indicator.
    """

    prevalence_data = {
        'underweight': compute_prevalence_rate_with_CI(wfa_series, threshold=-2, confidence=0.95),
        'severely_underweight': compute_prevalence_rate_with_CI(wfa_series, threshold=-3, confidence=0.95),
        'stunting': compute_prevalence_rate_with_CI(hfa_series, threshold=-2, confidence=0.95),
        'severely_stunting': compute_prevalence_rate_with_CI(hfa_series, threshold=-3, confidence=0.95),
        'thin': compute_prevalence_rate_with_CI(bmifa_series, threshold=-2, confidence=0.95),
        'overweight': compute_prevalence_rate_with_CI(over_60_mo_bmifa_series, threshold=1, above=True, confidence=0.95),
        'obese': compute_prevalence_rate_with_CI(over_60_mo_bmifa_series, threshold=2, above=True, confidence=0.95)
    }
    return prevalence_data

def display_text_below_axes(ax, text, loc, color=None):
    """
    Displays a text annotation below a set of axes. Optionally, displays a colored box next to the text.

    Args:
    - ax (Axes): The axes to display text below.
    - text (str): The text to display.
    - loc (int): The location index for the x-coordinate to display the text, based on predefined coordinates.
    - color (str, optional): The color of the box to display. Defaults to None.
    """

    text_art = ax.text(x=TEXT_X_COORDINATE[loc], y=-0.2, s=text, transform=ax.transAxes,
                       horizontalalignment='left', verticalalignment='top',
                       fontsize=10, color='black')
    if color:
        # Render the text to update its size information
        ax.figure.canvas.draw_idle()
        
        # Get and transform the bounding box of the text
        bbox = text_art.get_window_extent(renderer=ax.figure.canvas.get_renderer()).transformed(ax.transAxes.inverted())
        
        # Draw the colored box to the left of the text
        ax.add_patch(patches.Rectangle((bbox.x0 - 0.03, bbox.y0), 0.02, bbox.height,
                                       transform=ax.transAxes, color=color, clip_on=False))

def display_mean_below_axes(ax, mean, CI_tuple, loc):
    """
    Displays a mean value and its confidence interval below a set of axes.

    Args:
    - ax (Axis): The axes to display the mean below.
    - mean (float): The mean value to display.
    - CI_tuple (tuple[float, float]): The lower and upper bounds of the confidence interval of the mean.
    - loc (int): The location index for the x-coordinate to display the mean value, based on predefined coordinates.
    """

    text = f"Mean: {mean:.2f}\n95% CI: [{CI_tuple[0]:.2f}, {CI_tuple[1]:.2f}]"
    display_text_below_axes(ax, text, loc)

def display_all_means(axes, means, CI_tuples, loc):
    """
    Displays mean values and their confidence intervals below each set of axes. 

    Args:
    - axes (Iterable[Axes]): All of the Axes objects to be displayed under.
    - means (Iterable[float]): All of the mean values to be displayed.
    - CI_tuples (Iterable[Tuple[float, float]]): All of the confidence intervals to display
    - loc (int): The location index for the x-coordinate to display each mean value, based on predefined coordinates.
    """

    for ax, mean, CI_tuple in zip(axes, means, CI_tuples):
        display_mean_below_axes(ax, mean, CI_tuple, loc)

def display_prevalence_below_axes(ax, malnourishment_type, prevalence, CI_tuple, loc, color=None):
    """
    Displays a malnourishment prevalence rate and its confidence interval below a set of axes. Optionally, displays a colored box next to the prevalence rate.
    
    Args:
    - ax (Axes): The axes to display the prevalence rate below.
    - malnourishment_type (str): The type of malnourishment which prevalence of is being displayed.
    - prevalence (float): The prevalence rate to display.
    - CI_tuple (Tuple[float, float]): The lower and upper bounds of the confidence interval of the prevalence rate.
    - loc (int): The location index for the x-coordinate to display the prevalence rate, based on predefined coordinates.
    - color (str, optional): The color of the box to display. Defaults to None.
    """

    text = f"{malnourishment_type} prevalence: {prevalence:.1%}\n95% CI: [{CI_tuple[0]:.1%}, {CI_tuple[1]:.1%}]"
    display_text_below_axes(ax, text, loc, color)

def display_all_prevalences(prevalence_data, wfa_ax, hfa_ax, bmifa_ax, over_60_mo_bmifa_ax):
    """
    Displays the computed prevalences on the respective axes.

    Args:
    - prevalence_data (dict): A dictionary containing the prevalence data and confidence intervals.
    - wfa_ax, hfa_ax, bmifa_ax, over_60_mo_bmifa_ax (plt.Axes): Axes objects to display the prevalence data.
    """

    display_prevalence_below_axes(wfa_ax, "Underweight", *prevalence_data['underweight'], 1, COLORS['yellow'])
    display_prevalence_below_axes(wfa_ax, "Severely underweight", *prevalence_data['severely_underweight'], 2, COLORS['red'])
    display_prevalence_below_axes(hfa_ax, "Stunting", *prevalence_data['stunting'], 1, COLORS['yellow'])
    display_prevalence_below_axes(hfa_ax, "Severely stunting", *prevalence_data['severely_stunting'], 2, COLORS['red'])
    display_prevalence_below_axes(bmifa_ax, "Thin", *prevalence_data['thin'], 1, COLORS['yellow'])
    display_prevalence_below_axes(over_60_mo_bmifa_ax, "Overweight", *prevalence_data['overweight'], 1, COLORS['yellow'])
    display_prevalence_below_axes(over_60_mo_bmifa_ax, "Obese", *prevalence_data['obese'], 2, COLORS['red'])

def compute_and_display_analysis(wfa_series, hfa_series, bmifa_series, over_60_mo_bmifa_series, wfa_ax, hfa_ax, bmifa_ax, over_60_mo_bmifa_ax):
    """
    Computes and displays the analysis of anthropometric data including mean z-scores and malnourishment prevalences.

    Calculates the mean z-scores and their confidence intervals (CIs) for weight-for-age, height-for-age, 
    and BMI-for-age. Computes the prevalence rates of various malnourishment indicators along with their CIs.
    Results are displayed on the provided axes.

    Args:
    - wfa_series, hfa_series, bmifa_series, over_60_mo_bmifa_series (pd.Series): Series containing the anthropometric data.
    - wfa_ax, hfa_ax, bmifa_ax, over_60_mo_bmifa_ax (plt.Axes): Axes objects to display the analysis results.
    """

    # Compute mean z-scores
    wfa_mean, wfa_CI = compute_mean_with_CI(wfa_series, 0.95)
    hfa_mean, hfa_CI = compute_mean_with_CI(hfa_series, 0.95)
    bmifa_mean, bmifa_CI = compute_mean_with_CI(bmifa_series, 0.95)

    # Display means with CIs on figure
    display_all_means((wfa_ax, hfa_ax, bmifa_ax), (wfa_mean, hfa_mean, bmifa_mean), (wfa_CI, hfa_CI, bmifa_CI), 0)

    # Compute malnourishment prevalences
    prevalence_data = compute_prevalences(wfa_series, hfa_series, bmifa_series, over_60_mo_bmifa_series)

    # Display prevalences with CIs on figure
    display_all_prevalences(prevalence_data, wfa_ax, hfa_ax, bmifa_ax, over_60_mo_bmifa_ax)

def main():
    """
    Main function to orchestrate the data import, preparation, plotting, and analysis.

    This function manages the workflow of the script, which includes importing and preparing data, 
    creating figures and grid specifications for plotting, plotting the data, and performing as well
    as displaying the analysis of anthropometric indicators.
    """

    # Import and prepare data
    wfa_series, hfa_series, bmifa_series, over_60_mo_bmifa_series, age_series = import_and_prepare_data(DATA_FILE_NAME)

    # Create figures and gridspecs for plotting
    fig1, gs1, fig2, gs2 = create_figures()

    # Plot data on the created figures
    wfa_ax, hfa_ax, bmifa_ax, over_60_mo_bmifa_ax, age_ax = plot_data(wfa_series, hfa_series, bmifa_series, 
                                                                      over_60_mo_bmifa_series, age_series, 
                                                                      fig1, gs1, fig2, gs2)
    
    # Compute and display statistical analysis results
    compute_and_display_analysis(wfa_series, hfa_series, bmifa_series, over_60_mo_bmifa_series, 
                                      wfa_ax, hfa_ax, bmifa_ax, over_60_mo_bmifa_ax)

    # Apply style and show plotted figures
    sns.set_style('ticks')
    plt.show()


if __name__ == "__main__":
    main()