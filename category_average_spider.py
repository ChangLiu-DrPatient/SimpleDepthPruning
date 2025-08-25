# Category Average Spider Plot
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

# Calculate average performance for each category across all models
category_averages = {}
model_performances = {}

# Load results for each model
for model_name, results_path in models.items():
    try:
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        
        model_performances[model_name] = {}
        
        # Calculate performance for each category
        for category, challenge_list in challenges.items():
            category_performances = []
            for challenge in challenge_list:
                if challenge in results and 'acc_norm,none' in results[challenge]:
                    category_performances.append(results[challenge]['acc_norm,none'])
            
            if category_performances:
                avg_performance = np.mean(category_performances)
                model_performances[model_name][category] = avg_performance
                print(f"{model_name} - {category}: {avg_performance:.3f} (avg of {len(category_performances)} challenges)")
            else:
                print(f"Warning: No valid results for {model_name} - {category}")
                
    except FileNotFoundError:
        print(f"Warning: Results file not found for {model_name}")
        continue
    except Exception as e:
        print(f"Error loading {model_name}: {str(e)}")
        continue

# Calculate overall averages for each category
for category in challenges.keys():
    category_values = []
    for model_name in model_performances.keys():
        if category in model_performances[model_name]:
            category_values.append(model_performances[model_name][category])
    
    if category_values:
        category_averages[category] = np.mean(category_values)
        print(f"\nOverall average for {category}: {category_averages[category]:.3f}")

# Number of categories
N = len(category_averages)

# Compute angle for each axis
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # Complete the circle

# Create the plot
fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(projection='polar'))

# Track all performance values to set dynamic range
all_performance_values = []

# Plot each model's category averages
for model_idx, (model_name, model_data) in enumerate(model_performances.items()):
    # Get performance values for each category
    performance_values = []
    for category in challenges.keys():
        if category in model_data:
            performance_values.append(model_data[category])
        else:
            performance_values.append(0)  # Default value if category not found
    
    # Add the first value to complete the circle
    performance_values += performance_values[:1]
    all_performance_values.extend(performance_values[:-1])  # Exclude the repeated value
    
    # Plot the spider chart for this model (only lines, no fill)
    color = model_colors[model_idx % len(model_colors)]
    ax.plot(angles, performance_values, '-', linewidth=3, color=color, alpha=0.8, label=model_name)

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

# Set the labels (category names)
category_names = list(category_averages.keys())
ax.set_xticks(angles[:-1])
ax.set_xticklabels(category_names, size=12, fontweight='bold')

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
plt.title('Average Performance by Challenge Category Across All Models', size=18, y=1.08)

# Add legend for models
ax.legend(loc='lower right', bbox_to_anchor=(1.4, 0.2), 
          fontsize=16, title="Models", title_fontsize=16)

plt.tight_layout()
plt.show()