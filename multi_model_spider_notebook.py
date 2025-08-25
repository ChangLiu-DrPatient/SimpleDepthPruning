# Multi-Model Spider Plot for Jupyter Notebook
# Copy this code into your notebook cell

# Define the challenges dictionary
challenges = {
    "Mathematics and Logic": [
        "leaderboard_bbh_boolean_expressions",
        "leaderboard_bbh_temporal_sequences",
        "leaderboard_bbh_penguins_in_a_table",
        "leaderboard_bbh_reasoning_about_colored_objects",
        "leaderboard_bbh_tracking_shuffled_objects_three_objects",
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

# Define models and their results paths
models = {
    "base model": "/home/scratch/changl8/prune_llm/llama-2-7b-hf/none/results.pkl",
    "Shortgpt\n Shortened Llama": "/home/scratch/changl8/prune_llm/llama-2-7b-hf/shortgpt/3/results.pkl",
    # "Shortened Llama": "/home/scratch/changl8/prune_llm/llama-2-7b-hf/shortened_llm/1/results.pkl",
    "Laco": "/home/scratch/changl8/prune_llm/llama-2-7b-hf/laco/3/results.pkl",
    "SimpMergeAct": "/home/scratch/changl8/prune_llm/llama-2-7b-hf/merge_40_0.5/results.pkl",
}

# Define colors for different models
model_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Define colors for different challenge categories
category_colors = ['#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Collect all challenge names in order and assign category colors
all_challenges = []
challenge_categories = []
current_category = None
color_index = 0

for category, challenge_list in challenges.items():
    for challenge in challenge_list:
        all_challenges.append(challenge)
        challenge_categories.append(category)
        if category != current_category:
            current_category = category
            color_index = (color_index + 1) % len(category_colors)

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

# Number of variables
N = len(all_challenges)

# Compute angle for each axis
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # Complete the circle

# Create the plot
fig, ax = plt.subplots(figsize=(16, 16), subplot_kw=dict(projection='polar'))

# Track all performance values to set dynamic range
all_performance_values = []

# Plot each model
for model_idx, (model_name, results_path) in enumerate(models.items()):
    try:
        # Check if file exists
        import os
        if not os.path.exists(results_path):
            print(f"Warning: File not found for {model_name}: {results_path}")
            continue
            
        # Load results
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        
        print(f"Successfully loaded {model_name} with {len(results)} results")
        
        # Get performance values for each challenge
        performance_values = []
        missing_challenges = []
        for challenge in all_challenges:
            if challenge in results:
                if 'acc_norm,none' in results[challenge]:
                    performance_values.append(results[challenge]['acc_norm,none'])
                else:
                    print(f"Warning: 'acc_norm,none' not found in {challenge} for {model_name}")
                    performance_values.append(0)
            else:
                missing_challenges.append(challenge)
                performance_values.append(0)  # Default value if challenge not found
        
        if missing_challenges:
            print(f"Warning: Missing challenges for {model_name}: {missing_challenges[:3]}...")
        
        # Add the first value to complete the circle
        performance_values += performance_values[:1]
        all_performance_values.extend(performance_values[:-1])  # Exclude the repeated value
        
        # Plot the spider chart for this model (only lines, no fill)
        color = model_colors[model_idx % len(model_colors)]
        ax.plot(angles, performance_values, '-', linewidth=3, color=color, alpha=0.8, label=model_name)
        
        print(f"Plotted {model_name} with color {color}")
        
    except FileNotFoundError:
        print(f"Error: Results file not found for {model_name}: {results_path}")
        continue
    except Exception as e:
        print(f"Error loading {model_name}: {str(e)}")
        continue

# Set dynamic range based on maximum value across all models
if all_performance_values:
    max_performance = max(all_performance_values)
    min_performance = min(all_performance_values)
    
    # Exaggerate differences by using a smaller range
    performance_range = max_performance - min_performance
    if performance_range < 0.1:  # If differences are small
        # Use a smaller range to make differences more visible
        y_max = max_performance + 0.05
        y_min = max(0, min_performance - 0.05)
    else:
        # Round up to the nearest 0.1
        y_max = np.ceil(max_performance * 10) / 10
        y_min = 0
    
    ax.set_ylim(y_min, y_max)
    
    # Set custom y-ticks to get clean values
    custom_yticks = np.arange(y_min + 0.1, y_max + 0.1, 0.1)
    ax.set_yticks(custom_yticks)
    
    print(f"Performance range: {y_min:.2f} to {y_max:.2f} (max: {max_performance:.3f}, min: {min_performance:.3f})")
    print(f"Custom y-ticks: {custom_yticks}")
else:
    print("Warning: No performance values found!")

# Set the labels with reformatted names and category colors
clean_challenge_names = [reformat_challenge_name(name) for name in all_challenges]
ax.set_xticks(angles[:-1])
ax.set_xticklabels(clean_challenge_names, size=10)

# Color the labels by category
current_category = None
color_index = 0
for i, (label, category) in enumerate(zip(ax.get_xticklabels(), challenge_categories)):
    if category != current_category:
        current_category = category
        color_index = (color_index + 1) % len(category_colors)
    label.set_color(category_colors[color_index])
    label.set_weight('bold')

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

# Make y-axis tick labels appear along a horizontal line
# Get the current y-ticks and their positions
yticks = ax.get_yticks()
print(f"Final y-ticks: {yticks}")
ytick_positions = ax.get_yticklabels()

# Clear existing y-tick labels completely
ax.set_yticklabels([])

# Add horizontal text labels at the right side of the plot, positioned lower to avoid overlap
# Only add labels for unique tick values to avoid duplicates
unique_ticks = []
for tick_val in yticks:
    if tick_val > 0 and tick_val not in unique_ticks:  # Skip the center (0) and duplicates
        unique_ticks.append(tick_val)
        # Position the label at the right side of the plot (angle = 0), but lower
        x_pos = -0.1  # Move slightly to the left to avoid overlap with category labels
        y_pos = tick_val
        ax.text(x_pos, y_pos, f'{tick_val:.1f}', 
                ha='right', va='center', fontsize=16,
                transform=ax.transData)

# Add title
plt.title('Model Performance Comparison on Different Challenges', size=18, y=1.08)

# Add legend for models only
ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.0), 
          fontsize=16, title="Models", title_fontsize=16)

plt.tight_layout()
plt.show()

# Create separate figure for challenge categories legend
fig_legend, ax_legend = plt.subplots(figsize=(8, 6))
ax_legend.axis('off')

# Create legend entries for each category with colored text
legend_elements = []
for i, (category, color) in enumerate(zip(challenges.keys(), category_colors)):
    legend_elements.append(plt.Line2D([0], [0], color=color, linewidth=3, label=category))

# Create the legend
ax_legend.legend(handles=legend_elements, loc='center', fontsize=14, 
                title="Challenge Categories", title_fontsize=16)
ax_legend.set_title("Challenge Categories", fontsize=16, pad=20)

plt.tight_layout()
plt.show()

# Print performance summary by category for each model
print("\nPerformance Summary by Category:")
for model_name, results_path in models.items():
    try:
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        
        print(f"\n{model_name}:")
        for category, challenge_list in challenges.items():
            category_performances = []
            for challenge in challenge_list:
                if challenge in results:
                    category_performances.append(results[challenge]['acc_norm,none'])
            
            if category_performances:
                avg_performance = np.mean(category_performances)
                print(f"  {category}: {avg_performance:.3f} (avg of {len(category_performances)} challenges)")
    except FileNotFoundError:
        print(f"  Warning: Results file not found for {model_name}")
        continue 