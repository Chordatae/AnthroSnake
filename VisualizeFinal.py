# Import Statements
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import scipy.stats as stats
import math

# Constants:
Z_SCORE_TYPES = ("wfa_z_score", "hfa_z_score", "bmifa_z_score")
TEXT_X_COORDINATE = (-0.05, 0.23, 0.6)

# Function Definitions:
def isolate_series(data, category_to_isolate):
    """
    Isolate a series from a DataFrame.
    
    Args:
    - data (DataFrame): The DataFrame to isolate the series from.
    - category_to_isolate (str): The column name of the series to isolate.

    Returns:
    - Series: The isolated series.
    """

    return data[category_to_isolate]

def plot_z_distribution_kde(data, title, x_label, fig, position):
    """
    Plot a KDE distribution.
    
    Args:
    - data (Series): The data to plot.
    - title (str): The title of the plot.
    - x_label (str): The label for the x-axis.
    - fig (Figure): The figure to plot on.
    - position (int): The subplot position.

    Returns:
    - Axex: The Axes object of the plot.
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
    Shade a portion of a given distribution based on an x-value threshold.

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

def compute_mean_with_CI(data, confidence):
    """
    Compute the mean and confidence interval for a dataset.
    
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
    Compute the prevalence rate and confidence interval for a dataset.

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

def display_text_below_axes(ax, text, loc, color=None):
    """
    Display a text annotation below a set of axes. Optionally, display a colored box next to the text.

    Args:
    - ax (Axes): The axes to display text below.
    - text (str): The text to display.
    - loc (int): The location index for the x-coordinate to display the text, based on predefined coordinates.
    - color (str, optional): The color of the box to display. Defaults to None.
    """

    # First, draw the text to get its size
    text_art = ax.text(x=TEXT_X_COORDINATE[loc], y=-0.2, s=text, transform=ax.transAxes,
                       horizontalalignment='left', verticalalignment='top',
                       fontsize=10, color='black')
    if color:
        # Draw the canvas to ensure the text is rendered and its size is updated
        ax.figure.canvas.draw_idle()
        
        # Get the bounding box of the text
        bbox = text_art.get_window_extent(renderer=ax.figure.canvas.get_renderer())
        bbox = bbox.transformed(ax.transAxes.inverted())
        
        # Calculate the x position for the rectangle
        rect_x = bbox.x0 - 0.03  # Add a small left margin
        
        # Calculate the width and height of the rectangle
        rect_width = 0.02  # Fixed width for the colored box
        rect_height = bbox.height  # Height of the colored box
        
        # Draw the colored box to the left of the text
        ax.add_patch(patches.Rectangle((rect_x, bbox.y0), rect_width, rect_height,
                                       transform=ax.transAxes, color=color,
                                       clip_on=False))

def display_mean_below_axes(ax, mean, CI_tuple, loc):
    """
    Display a mean value and its confidence interval below a set of axes.

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
    Display mean values and their confidence intervals below each set of axes. 

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
    Display a malnourishment prevalence rate and its confidence interval below a set of axes. Optionally, display a colored box next to the prevalence rate.
    
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

# Indent body and uncomment this. Doesn't seem to work when running in interactive window in VS Code.
# def main():
# Import and prep data:
data_with_z_scores = pd.read_csv('output_with_z_scores.csv', index_col=0)

# Isolate data on kids over 60 months old, for overweight/obese prevalence.
data_over_60_months = data_with_z_scores[data_with_z_scores.age_months > 60].reset_index(drop=True)

#Create individual series object for each indicator (and extra for over 60 months BMIs)
all_z_score_series = {}
for z_score_type in Z_SCORE_TYPES:
    all_z_score_series[z_score_type] = isolate_series(data_with_z_scores, z_score_type)
all_z_score_series["over_60_months_bmifa_z_score"] = isolate_series(data_over_60_months, "bmifa_z_score")

#Create series for age:
age_series = isolate_series(data_with_z_scores, "age_months")

wfa_series = all_z_score_series["wfa_z_score"]
hfa_series = all_z_score_series["hfa_z_score"]
bmifa_series = all_z_score_series["bmifa_z_score"]
over_60_mo_bmifa_series = all_z_score_series["over_60_months_bmifa_z_score"]

#Trim subjects over 10 years old out of the wfa_series:
wfa_series = wfa_series[wfa_series.notna()].reset_index(drop=True)

#Plot data and compute aggregates/prevalence rates:
#Create the overall figures which the subplots will go on:
fig1 = plt.figure(figsize=(8, 15))
gs = gridspec.GridSpec(6, 1, height_ratios=[1, 0.3, 1, 0.3, 1, 0.3])
blank_ax = fig1.add_subplot(gs[5])
blank_ax.axis('off')

fig2 = plt.figure(figsize=(8, 9))
gs2 = gridspec.GridSpec(3, 1, height_ratios=[1, 0.3, 1])

#plot z-score distributions for each indicator:
wfa_ax = plot_z_distribution_kde(wfa_series, "Distribution of Weight-For-Age Z-Scores, Ages 0-10", "Z-score", fig1, gs[0])
hfa_ax = plot_z_distribution_kde(hfa_series, "Distribution of Height-For-Age Z-Scores", "Z-score", fig1, gs[2])
bmifa_ax = plot_z_distribution_kde(bmifa_series, "Distribution of BMI-For-Age Z-Scores", "Z-score", fig1, gs[4])
fig1_axes = (wfa_ax, hfa_ax, bmifa_ax)

over_60_mo_bmifa_ax = plot_z_distribution_kde(over_60_mo_bmifa_series, "Distribution of BMI-For-Age Z-Scores, Ages 5-19", "Z-score", fig2, gs2[0])

age_ax = fig2.add_subplot(gs2[2])
sns.kdeplot(age_series, ax=age_ax, fill=True)
age_ax.set_title("Distribution of Participant Ages")
age_ax.set_xlabel("Age (Months)")

#Shade prevalences of malnourishment:
shade_prevalence_on_distribution(wfa_series, wfa_ax, threshold=-2, color='#FFDFBA') #underweight
shade_prevalence_on_distribution(wfa_series, wfa_ax, threshold=-3, color='#FFB3BA') #severly underweight
shade_prevalence_on_distribution(hfa_series, hfa_ax, threshold=-2, color='#FFDFBA') #stunting
shade_prevalence_on_distribution(hfa_series, hfa_ax, threshold=-3, color='#FFB3BA') #severely stunting
shade_prevalence_on_distribution(bmifa_series, bmifa_ax, threshold=-2, color='#FFDFBA') #Thin
shade_prevalence_on_distribution(over_60_mo_bmifa_series, over_60_mo_bmifa_ax, threshold=1, above=True, color='#FFDFBA') #Overweight
shade_prevalence_on_distribution(over_60_mo_bmifa_series, over_60_mo_bmifa_ax, threshold=2, above=True, color='#FFB3BA') #Obese

#Compute mean z-scores:
wfa_mean, wfa_CI = compute_mean_with_CI(wfa_series, 0.95)
hfa_mean, hfa_CI = compute_mean_with_CI(hfa_series, 0.95)
bmifa_mean, bmifa_CI = compute_mean_with_CI(bmifa_series, 0.95)

all_means = (wfa_mean, hfa_mean, bmifa_mean)
all_mean_CIs = (wfa_CI, hfa_CI, bmifa_CI)

#Display means w/ CIs on figure:
display_all_means(fig1_axes, all_means, all_mean_CIs, 0)

#Compute malnourishment prevalences: 
p_underweight, p_underweight_CI = compute_prevalence_rate_with_CI(wfa_series, threshold=-2, confidence=0.95) #underweight
p_sev_underweight, p_sev_underweight_CI = compute_prevalence_rate_with_CI(wfa_series, threshold=-3, confidence=0.95) #severely underweight
p_stunting, p_stunting_CI = compute_prevalence_rate_with_CI(hfa_series, threshold=-2, confidence=0.95) #stunting
p_sev_stunting, p_sev_stunting_CI = compute_prevalence_rate_with_CI(hfa_series, threshold=-3, confidence=0.95) #severely stunting
p_thin, p_thin_CI = compute_prevalence_rate_with_CI(bmifa_series, threshold=-2, confidence=0.95) #thin
p_overweight, p_overweight_CI = compute_prevalence_rate_with_CI(over_60_mo_bmifa_series, threshold=1, above=True, confidence=0.95) #overweight
p_obese, p_obese_CI = compute_prevalence_rate_with_CI(over_60_mo_bmifa_series, threshold=2, above=True, confidence=0.95) #obese

# Display prevalences w/ CIs on figure:
# Note: some of these prevalences have thresholds (ie 2.5% is low, 5% is medium, etc. Don't know exact numbers). Figure them out and add them in.
display_prevalence_below_axes(wfa_ax, "Underweight", p_underweight, p_underweight_CI, 1, '#FFDFBA')
display_prevalence_below_axes(wfa_ax, "Severely underweight", p_sev_underweight, p_sev_underweight_CI, 2, '#FFB3BA')
display_prevalence_below_axes(hfa_ax, "Stunting", p_stunting, p_stunting_CI, 1, '#FFDFBA')
display_prevalence_below_axes(hfa_ax, "Severly stunting", p_sev_stunting, p_sev_stunting_CI, 2, '#FFB3BA')
display_prevalence_below_axes(bmifa_ax, "Thin", p_thin, p_thin_CI, 1, '#FFDFBA')
display_prevalence_below_axes(over_60_mo_bmifa_ax, "Overweight", p_overweight, p_overweight_CI, 1, '#FFDFBA')
display_prevalence_below_axes(over_60_mo_bmifa_ax, "Obese", p_obese, p_obese_CI, 2, '#FFB3BA')


#keeping these at the bottom:
#plt.tight_layout()
sns.set_style('ticks')
plt.show()




# if __name__ == "main":
#     main()