import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Define the challenges dictionary
challenges = {
    "Mathematics and Logic": [
        "leaderboard_bbh_boolean_expressions",
        "leaderboard_bbh_temporal_sequences",
        "leaderboard_bbh_penguins_in_a_table",
        "leaderboard_bbh_reasoning_about_colored_objects",
        "leaderboard_bbh_tracking_shuffled_objects_three_object",
        "leaderboard_bbh_logical_deduction_five_objects",
    ],
    "Natural Language Understanding": [
        "leaderboard_bbh_disambiguation_qa",
        "leaderboard_bbh_ruin_names",
        "leaderboard_bbh_salient_translation_error_detection",
    ],
    "Use of World Knowledge": [
        "leaderboard_bbh_movie_recommendation",
        "leaderboard_bbh_ruin_names",
        "leaderboard_bbh_sports_understanding",
    ],
    "Graduate Level Domain Knowledge": [
        "leaderboard_gpqa_main"
    ],
    "Multistep Soft Reasoning": [
        "leaderboard_musr_murder_mysteries",
        "leaderboard_musr_object_placements",
        "leaderboard_musr_team_allocation",
    ],
}

# Load results (you'll need to update this path)
results_path = '/home/scratch/changl8/prune_llm/llama-2-7b-hf/none/results.pkl'
with open(results_path, 'rb') as f:
    results = pickle.load(f)

# Define marker styles for different categories
marker_styles = ['o', 's', '^', 'D', 'v']  # circle, square, triangle up, diamond, triangle down
category_markers = {}

# Collect all challenge names in order and assign markers
all_challenges = []
challenge_categories = []
current_category = None
marker_index = 0

for category, challenge_list in challenges.items():
    for challenge in challenge_list:
        all_challenges.append(challenge)
        challenge_categories.append(category)
        if category != current_category:
            category_markers[category] = marker_styles[marker_index]
            marker_index = (marker_index + 1) % len(marker_styles)
            current_category = category

# Get performance values for each challenge
performance_values = []
for challenge in all_challenges:
    if challenge in results:
        performance_values.append(results[challenge]['acc_norm,none'])
    else:
        performance_values.append(0)  # Default value if challenge not found

# Number of variables
N = len(all_challenges)

# Compute angle for each axis
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # Complete the circle

# Add the first value to complete the circle
performance_values += performance_values[:1]

# Create the plot
fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(projection='polar'))

# Set dynamic range based on maximum value
max_performance = max(performance_values[:-1])  # Exclude the repeated first value
y_max = max_performance + 0.1
ax.set_ylim(0, y_max)

# Plot the spider chart with different markers for categories
ax.plot(angles, performance_values, '-', linewidth=2, color='#1f77b4', alpha=0.7)
ax.fill(angles, performance_values, color='#1f77b4', alpha=0.2)

# Plot individual points with different markers for each category (hollow black markers)
current_category = None
for i, (challenge, category, performance) in enumerate(zip(all_challenges, challenge_categories, performance_values[:-1])):
    marker = category_markers[category]
    if category != current_category:
        # Add to legend only for first occurrence of each category
        ax.plot(angles[i], performance, marker, color='black', markersize=10, 
                markerfacecolor='white', markeredgecolor='black', markeredgewidth=1.5,
                label=category)
        current_category = category
    else:
        # Don't add to legend for subsequent points in same category
        ax.plot(angles[i], performance, marker, color='black', markersize=10,
                markerfacecolor='white', markeredgecolor='black', markeredgewidth=1.5)

# Function to reformat challenge names
def reformat_challenge_name(name):
    # Remove "leaderboard_" prefix
    name = name.replace('leaderboard_', '')
    
    # Remove leading "bbh_" or "musr_"
    if name.startswith('bbh_'):
        name = name[4:]  # Remove "bbh_"
    elif name.startswith('musr_'):
        name = name[5:]  # Remove "musr_"
    
    # Replace underscores with spaces
    name = name.replace('_', ' ')
    
    # Split into words and add line break if more than 2 words
    words = name.split()
    if len(words) > 2:
        # Find a good break point (after 2 words)
        name = ' '.join(words[:2]) + '\n' + ' '.join(words[2:])
    
    return name

# Set the labels with reformatted names
clean_challenge_names = [reformat_challenge_name(name) for name in all_challenges]
ax.set_xticks(angles[:-1])
ax.set_xticklabels(clean_challenge_names, size=10)

# Adjust label positions to avoid overlap
for i, label in enumerate(ax.get_xticklabels()):
    angle = angles[i]
    # Adjust label position based on angle to avoid overlap
    if angle < np.pi/2 or angle > 3*np.pi/2:
        label.set_horizontalalignment('left')
    elif angle > np.pi/2 and angle < 3*np.pi/2:
        label.set_horizontalalignment('right')
    else:
        label.set_horizontalalignment('center')

# Add grid with larger font size for tick labels
ax.grid(True)
ax.tick_params(axis='y', labelsize=12)  # Make y-axis tick labels larger

# Add title
plt.title('Model Performance on Different Challenges', size=18, y=1.08)

# Add legend for model name only (positioned to avoid overlap with labels)
from matplotlib.patches import Patch
model_patch = Patch(color='#1f77b4', alpha=0.7, label='base model')
ax.legend(handles=[model_patch], loc='upper right', bbox_to_anchor=(1.4, 0.8), 
          fontsize=12, title="Model", title_fontsize=12)

plt.tight_layout()
plt.show()

# Create separate figure for challenge categories legend
fig_legend, ax_legend = plt.subplots(figsize=(8, 6))
ax_legend.axis('off')

# Create legend entries for each category
legend_elements = []
for category, marker in category_markers.items():
    legend_elements.append(plt.Line2D([0], [0], marker=marker, color='black', 
                                     markersize=12, markerfacecolor='white', 
                                     markeredgecolor='black', markeredgewidth=1.5, 
                                     label=category, linestyle=''))

# Create the legend
ax_legend.legend(handles=legend_elements, loc='center', fontsize=14, 
                title="Challenge Categories", title_fontsize=16)
ax_legend.set_title("Challenge Categories", fontsize=16, pad=20)

plt.tight_layout()
plt.show()

# Print performance summary by category
print("\nPerformance Summary by Category:")
for category, challenge_list in challenges.items():
    category_performances = []
    for challenge in challenge_list:
        if challenge in results:
            category_performances.append(results[challenge]['acc_norm,none'])
    
    if category_performances:
        avg_performance = np.mean(category_performances)
        print(f"{category}: {avg_performance:.3f} (avg of {len(category_performances)} challenges)") 