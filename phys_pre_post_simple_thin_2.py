import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_pre_post_data(csv_file_path, sampling_rate=100.0, pre_duration=5.0, post_duration=5.0):
    """
    Analyze pre/post data from CSV file where each column contains combined pre+post data
    
    Parameters:
    - csv_file_path: path to CSV file
    - sampling_rate: data sampling rate in Hz (default 100 Hz = 0.01s intervals)
    - pre_duration: duration of pre-sound period in seconds (default 5.0s)
    - post_duration: duration of post-sound period in seconds (default 5.0s)
    """
    
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    print("Data shape:", df.shape)
    print("\nColumn names:")
    print(df.columns.tolist())
    print("\nFirst few rows:")
    print(df.head())
    
    # Calculate expected data points
    dt = 1.0 / sampling_rate  # time interval between samples
    pre_points = int(pre_duration / dt)
    post_points = int(post_duration / dt)
    total_expected = pre_points + post_points
    
    print(f"\nData format parameters:")
    print(f"  Sampling rate: {sampling_rate} Hz (dt = {dt:.4f}s)")
    print(f"  Pre-sound duration: {pre_duration}s ({pre_points} points)")
    print(f"  Post-sound duration: {post_duration}s ({post_points} points)")
    print(f"  Expected total points per subject: {total_expected}")
    print(f"  Actual rows in data: {len(df)}")
    
    # Find subject columns (exclude time columns)
    time_columns = [col for col in df.columns if 'time' in col.lower()]
    subject_columns = [col for col in df.columns if col not in time_columns]
    
    print(f"\nTime columns: {time_columns}")
    print(f"Subject columns ({len(subject_columns)}): {subject_columns}")
    
    # Handle time axis
    if time_columns:
        time_col = time_columns[0]
        time_values = df[time_col]
        print(f"Using time column: {time_col}")
        print(f"Time range: {time_values.min():.3f} to {time_values.max():.3f} seconds")
        
        # Check if time starts at 0 or needs adjustment
        if time_values.min() >= 0:
            # Time probably starts at 0, sound onset is at pre_duration
            sound_onset_time = pre_duration
        else:
            # Time might be centered around 0, find the midpoint
            sound_onset_time = (time_values.max() + time_values.min()) / 2
        
        print(f"Inferred sound onset time: {sound_onset_time:.3f}s")
    else:
        # Create time column based on row index
        time_values = df.index * dt
        sound_onset_time = pre_duration
        print(f"Created time column based on {sampling_rate} Hz sampling")
        print(f"Sound onset at: {sound_onset_time:.3f}s")
    
    # Define analysis windows: 2.8s on each side of stimulus
    analysis_duration = 0.5  # seconds - CHANGED FROM 4
    analysis_points = int(analysis_duration / dt)
    
    # Define display windows: 4.5s on each side of stimulus
    display_duration = 4.5  # seconds
    display_points = int(display_duration / dt)
    
    # Pre analysis window: from (sound_onset - 2.8s) to sound_onset
    pre_start_time = sound_onset_time - analysis_duration
    pre_end_time = sound_onset_time
    
    # Post analysis window: from sound_onset to (sound_onset + 2.8s)
    post_start_time = sound_onset_time
    post_end_time = sound_onset_time + analysis_duration
    
    # Pre display window: from (sound_onset - 4.5s) to sound_onset
    pre_display_start_time = sound_onset_time - display_duration
    pre_display_end_time = sound_onset_time
    
    # Post display window: from sound_onset to (sound_onset + 4.5s)
    post_display_start_time = sound_onset_time
    post_display_end_time = sound_onset_time + display_duration
    
    print(f"\nAnalysis windows ({analysis_duration}s each, {analysis_points} points):")
    print(f"  Pre window: {pre_start_time:.3f}s to {pre_end_time:.3f}s")
    print(f"  Post window: {post_start_time:.3f}s to {post_end_time:.3f}s")
    
    print(f"\nDisplay windows ({display_duration}s each, {display_points} points):")
    print(f"  Pre display: {pre_display_start_time:.3f}s to {pre_display_end_time:.3f}s")
    print(f"  Post display: {post_display_start_time:.3f}s to {post_display_end_time:.3f}s")
    
    # Create masks for analysis windows
    pre_mask = (time_values >= pre_start_time) & (time_values < pre_end_time)
    post_mask = (time_values >= post_start_time) & (time_values < post_end_time)
    
    # Create masks for display windows
    pre_display_mask = (time_values >= pre_display_start_time) & (time_values < pre_display_end_time)
    post_display_mask = (time_values >= post_display_start_time) & (time_values < post_display_end_time)
    
    print(f"\nData points in analysis windows:")
    print(f"  Pre window: {pre_mask.sum()} points")
    print(f"  Post window: {post_mask.sum()} points")
    
    print(f"\nData points in display windows:")
    print(f"  Pre display: {pre_display_mask.sum()} points")
    print(f"  Post display: {post_display_mask.sum()} points")
    
    if pre_mask.sum() == 0 or post_mask.sum() == 0:
        print("ERROR: No data found in analysis windows. Check time alignment.")
        return {}
    
    # Extract data for each subject (both analysis and display windows)
    pre_data_subjects = []
    post_data_subjects = []
    pre_display_data_subjects = []
    post_display_data_subjects = []
    valid_subjects = []
    
    for col in subject_columns:
        if col in df.columns:
            # Extract analysis data for this subject
            pre_subject_data = df.loc[pre_mask, col].dropna()
            post_subject_data = df.loc[post_mask, col].dropna()
            
            # Extract display data for this subject
            pre_display_subject_data = df.loc[pre_display_mask, col].dropna()
            post_display_subject_data = df.loc[post_display_mask, col].dropna()
            
            # Check if we have enough data points for analysis
            if len(pre_subject_data) > 10 and len(post_subject_data) > 10:
                pre_data_subjects.append(pre_subject_data.values)
                post_data_subjects.append(post_subject_data.values)
                pre_display_data_subjects.append(pre_display_subject_data.values)
                post_display_data_subjects.append(post_display_subject_data.values)
                valid_subjects.append(col)
                print(f"  {col}: Analysis Pre={len(pre_subject_data)} pts, Post={len(post_subject_data)} pts")
                print(f"       Display Pre={len(pre_display_subject_data)} pts, Post={len(post_display_subject_data)} pts")
            else:
                print(f"  {col}: SKIPPED (insufficient data)")
    
    n_subjects = len(valid_subjects)
    print(f"\nValid subjects for analysis: {n_subjects}")
    
    if n_subjects == 0:
        print("ERROR: No subjects with sufficient data for analysis")
        return {}
    
    # Find minimum length to ensure all subjects have same number of points
    min_len_pre = min(len(data) for data in pre_data_subjects)
    min_len_post = min(len(data) for data in post_data_subjects)
    min_len_pre_display = min(len(data) for data in pre_display_data_subjects)
    min_len_post_display = min(len(data) for data in post_display_data_subjects)
    
    print(f"Using {min_len_pre} points for pre analysis, {min_len_post} points for post analysis")
    print(f"Using {min_len_pre_display} points for pre display, {min_len_post_display} points for post display")
    
    # Truncate all subjects to minimum length and convert to array
    pre_array = np.array([data[:min_len_pre] for data in pre_data_subjects])
    post_array = np.array([data[:min_len_post] for data in post_data_subjects])
    pre_display_array = np.array([data[:min_len_pre_display] for data in pre_display_data_subjects])
    post_display_array = np.array([data[:min_len_post_display] for data in post_display_data_subjects])
    
    # Calculate average across subjects
    pre_avg = np.mean(pre_array, axis=0)
    post_avg = np.mean(post_array, axis=0)
    pre_display_avg = np.mean(pre_display_array, axis=0)
    post_display_avg = np.mean(post_display_array, axis=0)


    
    
    # Also calculate standard error across subjects for each time point
    pre_sem = np.std(pre_array, axis=0) / np.sqrt(n_subjects)
    post_sem = np.std(post_array, axis=0) / np.sqrt(n_subjects)
    pre_display_sem = np.std(pre_display_array, axis=0) / np.sqrt(n_subjects)
    post_display_sem = np.std(post_display_array, axis=0) / np.sqrt(n_subjects)
    
    print(f"\nAveraged analysis data shapes:")
    print(f"Pre average: {pre_avg.shape}")
    print(f"Post average: {post_avg.shape}")
    
    print(f"\nAveraged display data shapes:")
    print(f"Pre display average: {pre_display_avg.shape}")
    print(f"Post display average: {post_display_avg.shape}")
    
   
    
    # Calculate descriptive statistics (only on analysis windows)
    pre_stats = {
        'mean': np.mean(pre_avg),
        'std': np.std(pre_avg),
        'median': np.median(pre_avg),
        'min': np.min(pre_avg),
        'max': np.max(pre_avg),
        'sem_mean': np.mean(pre_sem)  # Average standard error across time points
    }
    
    post_stats = {
        'mean': np.mean(post_avg),
        'std': np.std(post_avg),
        'median': np.median(post_avg),
        'min': np.min(post_avg),
        'max': np.max(post_avg),
        'sem_mean': np.mean(post_sem)  # Average standard error across time points
    }
    
    print(f"\nPre stats (averaged): Mean={pre_stats['mean']:.4f}, Std={pre_stats['std']:.4f}")
    print(f"Post stats (averaged): Mean={post_stats['mean']:.4f}, Std={post_stats['std']:.4f}")
    print(f"Average SEM - Pre: {pre_stats['sem_mean']:.4f}, Post: {post_stats['sem_mean']:.4f}")
    
    # Statistical tests on averaged data
    # 1. Paired t-test (since we have same time points)
    if len(pre_avg) == len(post_avg):
        t_stat_paired, p_val_paired = stats.ttest_rel(pre_avg, post_avg)
        print(f"\nPaired t-test (on averaged data): t={t_stat_paired:.4f}, p={p_val_paired:.6f}")
    else:
        t_stat_paired, p_val_paired = None, None
        print(f"\nCannot perform paired t-test (different lengths: {len(pre_avg)} vs {len(post_avg)})")
    
    # 2. Independent t-test
    t_stat_ind, p_val_ind = stats.ttest_ind(pre_avg, post_avg)
    print(f"Independent t-test (on averaged data): t={t_stat_ind:.4f}, p={p_val_ind:.6f}")
    
    # 3. Mann-Whitney U test (non-parametric)
    u_stat, p_val_mw = stats.mannwhitneyu(pre_avg, post_avg, alternative='two-sided')
    print(f"Mann-Whitney U test: U={u_stat:.4f}, p={p_val_mw:.6f}")
    
    # 4. Kolmogorov-Smirnov test
    ks_stat, p_val_ks = stats.ks_2samp(pre_avg, post_avg)
    print(f"Kolmogorov-Smirnov test: KS={ks_stat:.4f}, p={p_val_ks:.6f}")
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(pre_avg) - 1) * pre_stats['std']**2 + 
                         (len(post_avg) - 1) * post_stats['std']**2) / 
                        (len(pre_avg) + len(post_avg) - 2))
    cohens_d = (post_stats['mean'] - pre_stats['mean']) / pooled_std
    print(f"Effect size (Cohen's d): {cohens_d:.4f}")
    
    # Additional analysis: Subject-level statistics (using analysis windows only)
    # Calculate mean for each subject in pre and post periods
    pre_subject_means = [np.mean(data[:min_len_pre]) for data in pre_data_subjects]
    post_subject_means = [np.mean(data[:min_len_post]) for data in post_data_subjects]
    
    # Subject-level paired analysis
    print(f"\n--- Subject-level analysis (n={n_subjects} subjects) ---")
    t_stat_subjects, p_val_subjects = stats.ttest_rel(pre_subject_means, post_subject_means)
    print(f"Paired t-test on subject means: t={t_stat_subjects:.4f}, p={p_val_subjects:.6f}")
    
    # Effect size for subject-level comparison
    diff_means = np.array(post_subject_means) - np.array(pre_subject_means)
    cohens_d_subjects = np.mean(diff_means) / np.std(diff_means) if np.std(diff_means) > 0 else 0
    print(f"Subject-level effect size: {cohens_d_subjects:.4f}")
    
    print(f"Mean change per subject: {np.mean(diff_means):.4f} ± {np.std(diff_means):.4f}")
    print(f"Subject means - Pre: {np.mean(pre_subject_means):.4f} ± {np.std(pre_subject_means):.4f}")
    print(f"Subject means - Post: {np.mean(post_subject_means):.4f} ± {np.std(post_subject_means):.4f}")
    
    # Store results
    results = {}
    results["analysis_params"] = {
        'sampling_rate': sampling_rate,
        'pre_duration': pre_duration,
        'post_duration': post_duration,
        'analysis_duration': analysis_duration,
        'display_duration': display_duration,
        'sound_onset_time': sound_onset_time,
        'pre_window': (pre_start_time, pre_end_time),
        'post_window': (post_start_time, post_end_time),
        'pre_display_window': (pre_display_start_time, pre_display_end_time),
        'post_display_window': (post_display_start_time, post_display_end_time),
        'time_values': time_values,
        'pre_mask': pre_mask,
        'post_mask': post_mask,
        'pre_display_mask': pre_display_mask,
        'post_display_mask': post_display_mask,
        'dt': dt
    }
    
    results["averaged_analysis"] = {
        'pre_stats': pre_stats,
        'post_stats': post_stats,
        'pre_data': pre_avg,
        'post_data': post_avg,
        'pre_display_data': pre_display_avg,
        'post_display_data': post_display_avg,
        'pre_sem': pre_sem,
        'post_sem': post_sem,
        'pre_display_sem': pre_display_sem,
        'post_display_sem': post_display_sem,
        'pre_array': pre_array,  # Individual subject data (analysis)
        'post_array': post_array,  # Individual subject data (analysis)
        'pre_display_array': pre_display_array,  # Individual subject data (display)
        'post_display_array': post_display_array,  # Individual subject data (display)
        'valid_subjects': valid_subjects,
        'tests': {
            'paired_t': (t_stat_paired, p_val_paired),
            'independent_t': (t_stat_ind, p_val_ind),
            'mann_whitney': (u_stat, p_val_mw),
            'kolmogorov_smirnov': (ks_stat, p_val_ks),
            'cohens_d': cohens_d,
            'subject_paired_t': (t_stat_subjects, p_val_subjects),
            'cohens_d_subjects': cohens_d_subjects
        },
        'n_subjects': n_subjects
    }
    
    # Significance interpretation
    alpha = 0.05
    print(f"\n=== SIGNIFICANCE SUMMARY (α = {alpha}) ===")
    
    # Time-series level analysis
    if p_val_ind < alpha:
        print(f"✓ SIGNIFICANT difference in averaged time-series (p = {p_val_ind:.6f})")
    else:
        print(f"✗ No significant difference in averaged time-series (p = {p_val_ind:.6f})")
    
    # Subject-level analysis
    if p_val_subjects < alpha:
        print(f"✓ SIGNIFICANT difference at subject level (p = {p_val_subjects:.6f})")
    else:
        print(f"✗ No significant difference at subject level (p = {p_val_subjects:.6f})")
    
    # Effect sizes
    if abs(cohens_d) > 0.8:
        print(f"✓ Large effect size for time-series (|d| = {abs(cohens_d):.4f})")
    elif abs(cohens_d) > 0.5:
        print(f"~ Medium effect size for time-series (|d| = {abs(cohens_d):.4f})")
    elif abs(cohens_d) > 0.2:
        print(f"~ Small effect size for time-series (|d| = {abs(cohens_d):.4f})")
    else:
        print(f"- Negligible effect size for time-series (|d| = {abs(cohens_d):.4f})")
    
    if abs(cohens_d_subjects) > 0.8:
        print(f"✓ Large effect size for subjects (|d| = {abs(cohens_d_subjects):.4f})")
    elif abs(cohens_d_subjects) > 0.5:
        print(f"~ Medium effect size for subjects (|d| = {abs(cohens_d_subjects):.4f})")
    elif abs(cohens_d_subjects) > 0.2:
        print(f"~ Small effect size for subjects (|d| = {abs(cohens_d_subjects):.4f})")
    else:
        print(f"- Negligible effect size for subjects (|d| = {abs(cohens_d_subjects):.4f})")
    
    # Create visualization
    create_comparison_plots(results)
    
    return results

def create_continuous_trace(pre_data, post_data, dt, baseline_align=True, smooth_params=None):
    """
    Create a continuous, smooth trace from pre and post data segments.
    
    Parameters:
    - pre_data: array of pre-stimulus data
    - post_data: array of post-stimulus data  
    - dt: time interval between samples
    - baseline_align: if True, align the traces at t=0 for continuity
    - smooth_params: dict with 'window' and 'polyorder' for smoothing
    
    Returns:
    - combined_times: time axis for full trace
    - combined_smooth: smoothed continuous trace
    - pre_times, post_times: individual time axes
    - pre_smooth, post_smooth: individual smoothed segments
    """
    
    if smooth_params is None:
        smooth_params = {'window': 2, 'polyorder': 1}
    
    # Create time axes relative to sound onset (t=0)
    pre_times = np.linspace(-len(pre_data)*dt, -dt, len(pre_data))
    post_times = np.linspace(0, (len(post_data)-1)*dt, len(post_data))
    
    # Optional: Baseline alignment for continuity
    if baseline_align:
        # Align traces by adjusting the offset at t=0
        # Use last few points of pre and first few points of post to estimate baseline shift
        n_baseline = min(20, len(pre_data)//4, len(post_data)//4)
        pre_end_mean = np.mean(pre_data[-n_baseline:]) if n_baseline > 0 else pre_data[-1]
        post_start_mean = np.mean(post_data[:n_baseline]) if n_baseline > 0 else post_data[0]
        baseline_shift = post_start_mean - pre_end_mean
        
        print(f"Baseline alignment: shifting post data by {-baseline_shift:.4f} for continuity")
        post_data_aligned = post_data - baseline_shift
    else:
        post_data_aligned = post_data.copy()
        baseline_shift = 0
    
    # Combine data for continuous smoothing
    combined_data = np.concatenate([pre_data, post_data_aligned])
    combined_times = np.concatenate([pre_times, post_times])
    
    # Smooth the entire combined trace
    window_length = min(smooth_params['window'], len(combined_data) // 2 * 2 + 1)
    if window_length < 5:
        window_length = 1
    polyorder = min(smooth_params['polyorder'], window_length - 1)
    
    print(f"Smoothing continuous trace: {len(combined_data)} points, window={window_length} ({window_length*dt:.3f}s)")
    
    #combined_smooth = savgol_filter(combined_data, window_length, polyorder)
    combined_smooth = combined_data
    
    # Split back into pre and post for individual plotting
    n_pre = len(pre_data)
    pre_smooth = combined_smooth[:n_pre]
    post_smooth = combined_smooth[n_pre:]
    
    return {
        'combined_times': combined_times,
        'combined_data': combined_data,
        'combined_smooth': combined_smooth,
        'pre_times': pre_times,
        'post_times': post_times,
        'pre_smooth': pre_smooth,
        'post_smooth': post_smooth,
        'baseline_shift': baseline_shift
    }

def create_comparison_plots(results):
    """Create comparison plots with smooth continuous traces"""
    
    if 'averaged_analysis' not in results:
        print("No averaged analysis data to plot")
        return
    
    data = results['averaged_analysis']
    params = results['analysis_params']
    
    # Use analysis data for statistics (2.8s windows)
    pre_avg = data['pre_data']
    post_avg = data['post_data']
    pre_sem = data['pre_sem']
    post_sem = data['post_sem']
    pre_array = data['pre_array']
    post_array = data['post_array']
    
    # Use display data for plotting (4.5s windows)
    pre_display_avg = data['pre_display_data']
    post_display_avg = data['post_display_data']
    pre_display_sem = data['pre_display_sem']
    post_display_sem = data['post_display_sem']
    pre_display_array = data['pre_display_array']
    post_display_array = data['post_display_array']
    
    n_subjects = data['n_subjects']
    dt = params['dt']
    analysis_duration = params['analysis_duration']
    display_duration = params['display_duration']
    
    # Create continuous smooth trace for group average (using DISPLAY data)
    group_trace = create_continuous_trace(pre_display_avg, post_display_avg, dt, baseline_align=True, 
                                        smooth_params={'window': 9, 'polyorder': 3})
    
    # Create continuous traces for each individual subject (using DISPLAY data)
    individual_traces = []
    for i in range(min(n_subjects, len(pre_display_array), len(post_display_array))):
        if i < len(pre_display_array) and i < len(post_display_array):
            trace = create_continuous_trace(pre_display_array[i], post_display_array[i], dt, baseline_align=True,
                                          smooth_params={'window': 31, 'polyorder': 3})
            individual_traces.append(trace)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Box plot comparison (using ANALYSIS data for statistics)
    ax1 = axes[0, 0]
    box_data = [pre_avg, post_avg]
    labels = [f'Pre Sound\n( {analysis_duration}s)', f'Post Sound\n( {analysis_duration}s)']
    bp = ax1.boxplot(box_data, labels=labels, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax1.set_title(f'Pre vs Post Sound Comparison\n(Analysis: {analysis_duration}s windows, n={n_subjects} subjects)')
    ax1.set_ylabel('Δ F / F')
    ax1.grid(True, alpha=0.3)
    
    # 2. CONTINUOUS smooth time series plot (DISPLAY data with analysis windows highlighted)
    ax2 = axes[0, 1]
    
    # Plot the continuous smooth trace
    ax2.plot(group_trace['combined_times'], group_trace['combined_smooth'], 
             'darkblue', linewidth=1, label='', zorder=3)
    
    # Show raw data as thin line for reference
    ax2.plot(group_trace['combined_times'], group_trace['combined_data'], 
             'darkblue', alpha=0.3, linewidth=0.8, label='Raw Data', zorder=1)
    
    # Create continuous SEM bands
    pre_times = group_trace['pre_times']
    post_times = group_trace['post_times']
    combined_times = group_trace['combined_times']
    
    # Align post SEM with the same baseline shift
    combined_sem = np.concatenate([pre_display_sem, post_display_sem])
    combined_smooth = group_trace['combined_smooth']
    
    # Smooth the SEM as well for continuity
    window_length = min(61, len(combined_sem) // 2 * 2 + 1)
    if window_length >= 5:
        polyorder = min(3, window_length - 1)
        combined_sem_smooth = savgol_filter(combined_sem, window_length, polyorder)
    else:
        combined_sem_smooth = combined_sem
    
    ax2.fill_between(combined_times, 
                     combined_smooth - combined_sem_smooth, 
                     combined_smooth + combined_sem_smooth,
                     color='darkblue', alpha=0.2, zorder=2)
    
    # Add sound onset line
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.8, 
                label='SOUND ONSET', zorder=4)
    
    # Add ANALYSIS windows as gray shading
    ax2.axvspan(-analysis_duration, 0, alpha=0.15, color='gray', label=f'Pre Analysis ({analysis_duration}s)', zorder=0)
    ax2.axvspan(0, analysis_duration, alpha=0.15, color='gray', label=f'Post Analysis ({analysis_duration}s)', zorder=0)
    
    # Add display windows as subtle lines
    ax2.axvline(x=-display_duration, color='lightgray', linestyle=':', alpha=0.7, label=f'Display Window ({display_duration}s)')
    ax2.axvline(x=display_duration, color='lightgray', linestyle=':', alpha=0.7)
    
    ax2.set_xlabel('Time relative to sound (s)')
    ax2.set_ylabel('Δ F / F (mean ± SEM)')
    ax2.set_title(f'Continuous Signal Trace (Display: ±{display_duration}s, Analysis: ±{analysis_duration}s)\n(n={n_subjects} subjects, baseline-aligned)')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Set x-axis limits to show full display window
    ax2.set_xlim(-display_duration, display_duration)
    
    # 3. Individual subject continuous traces (DISPLAY data with analysis windows)
    ax3 = axes[1, 0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(individual_traces)))
    
    # Calculate y-axis limits based on all individual traces
    all_y_values = []
    for trace in individual_traces:
        all_y_values.extend(trace['combined_smooth'])
    y_min, y_max = np.min(all_y_values), np.max(all_y_values)
    y_range = y_max - y_min
    y_margin = y_range * 0.05  # 5% margin
    global_ylim = (-1 - y_margin, y_max + y_margin)
    
    for i, trace in enumerate(individual_traces):
        ax3.plot(trace['combined_times'], trace['combined_smooth'], 
                color=colors[i], alpha=0.6, linewidth=0.8, 
                label=f'Subject {i+1}' if i < 5 else "")  # Only label first 5
    
    # Overlay group average as thick line
    ax3.plot(group_trace['combined_times'], group_trace['combined_smooth'], 
             'black', linewidth=1, label='Group Average', zorder=10)
    
    # Add analysis windows as gray shading
    ax3.axvspan(-analysis_duration, 0, alpha=0.15, color='gray', zorder=0)
    ax3.axvspan(0, analysis_duration, alpha=0.15, color='gray', zorder=0)
    
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.8, 
                label='SOUND ONSET', zorder=11)
    
    ax3.set_xlabel('Time relative to sound (s)')
    ax3.set_ylabel('Δ F / F')
    ax3.set_title(f'Individual Traces + Group Average (Display: ±{display_duration}s)\nGray = Analysis Windows (±{analysis_duration}s)')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-display_duration, display_duration)
    ax3.set_ylim(global_ylim)
    
    # 4. Subject-level means comparison (using ANALYSIS data)
    ax4 = axes[1, 1]
    pre_means = [np.mean(subj_data) for subj_data in pre_array]
    post_means = [np.mean(subj_data) for subj_data in post_array]
    
    for i in range(len(pre_means)):
        color = 'green' if post_means[i] > pre_means[i] else 'red'
        ax4.plot([1, 2], [pre_means[i], post_means[i]], 'o-', 
                color=color, alpha=0.6, linewidth=2, label=f'Subject {i+1}' if i < 3 else "")
    
    bp = ax4.boxplot([pre_means, post_means], labels=[f'Pre Sound\n({analysis_duration}s)', f'Post Sound\n({analysis_duration}s)'], 
                    positions=[1, 2], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    
    ax4.set_title(f'Subject-Level Mean Changes\n(Analysis windows: {analysis_duration}s, n={len(pre_means)} subjects)')
    ax4.set_ylabel('Mean Δ F / F')
    ax4.set_xlim(0.5, 2.5)
    ax4.grid(True, alpha=0.3)
    
    mean_change = np.mean(np.array(post_means) - np.array(pre_means))
    ax4.text(1.5, ax4.get_ylim()[0] + 0.1 * (ax4.get_ylim()[1] - ax4.get_ylim()[0]), 
            f'Mean change: {mean_change:.4f}', ha='center', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle(f'Pre/Post Sound Analysis - Analysis: ±{analysis_duration}s, Display: ±{display_duration}s', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Create individual subject grid plot with continuous traces
    create_individual_continuous_plots(results, individual_traces, group_trace, global_ylim)

def create_individual_continuous_plots(results, individual_traces, group_trace, global_ylim):
    """Create individual subject plots with continuous traces"""
    
    data = results['averaged_analysis']
    params = results['analysis_params']
    n_subjects = data['n_subjects']
    analysis_duration = params['analysis_duration']
    display_duration = params['display_duration']
    
    # Grid layout for individual subjects
    n_cols = min(2, n_subjects)
    n_rows = (n_subjects + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    if n_subjects == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    axes_flat = axes.flatten() if n_subjects > 1 else axes
    
    for i in range(min(n_subjects, len(individual_traces))):
        ax = axes_flat[i] if n_subjects > 1 else axes_flat[0]
        trace = individual_traces[i]
        
        # Plot continuous smooth trace
        ax.plot(trace['combined_times'], trace['combined_smooth'], 
               'darkblue', linewidth=1.0)
        
        # Add analysis windows as gray shading
        ax.axvspan(-analysis_duration, 0, alpha=0.15, color='gray', zorder=0)
        ax.axvspan(0, analysis_duration, alpha=0.15, color='gray', zorder=0)
        
        # Add sound onset
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
        
        subject_name = f'{i+1}'  # Just the number
        ax.set_title(f'{subject_name}', fontsize = 4)
        if i == 0:  # Only on the first plot
            ax.set_xlabel('Time relative to sound (s)', fontsize=8)
            ax.set_ylabel('Δ F / F')
        else:
            ax.set_xticklabels([]) 
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-display_duration, display_duration)
        ax.set_ylim(-2,2)  # Use same y-axis scale for all subjects

   
        
        # Only show legend on first subplot
        if i == 0:
            ax.legend(loc='best', fontsize=8)
    
    # Hide unused subplots
    for i in range(len(individual_traces), len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.suptitle(f'Individual Subject Traces: Display ±{display_duration}s, Gray = Analysis Windows ±{analysis_duration}s\n(n={n_subjects} subjects, same y-axis scale)', 
                 fontsize=8, fontweight='bold')
    plt.tight_layout(pad=3)
    plt.show()
    
    # Create overlay plot showing all subjects + group average
    fig2, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Plot individual subjects as thin lines
    colors = plt.cm.viridis(np.linspace(0, 1, len(individual_traces)))
    for i, trace in enumerate(individual_traces):
        if i == 0:  # Only on the first plot
            subject_name = f'{i+1}'
            ax.set_title(f'{subject_name}', fontsize=10)
            ax.set_xlabel('Time relative to sound (s)', fontsize=8)
            ax.set_ylabel('Δ F / F', fontsize=8)
    
    # Plot group average as thick black line
    ax.plot(group_trace['combined_times'], group_trace['combined_smooth'], 
           'black', linewidth=1, label='Group Average', zorder=10)
    
    # Add confidence band for group average
    combined_sem = np.concatenate([data['pre_display_sem'], data['post_display_sem']])
    window_length = min(61, len(combined_sem) // 2 * 2 + 1)
    if window_length >= 5:
        polyorder = min(3, window_length - 1)
        combined_sem_smooth = savgol_filter(combined_sem, window_length, polyorder)
        
        ax.fill_between(group_trace['combined_times'], 
                       group_trace['combined_smooth'] - combined_sem_smooth,
                       group_trace['combined_smooth'] + combined_sem_smooth,
                       color='black', alpha=0.2, zorder=9, label='Group SEM')
    
    # Add sound onset line
    ax.axvline(x=0, color='red', linestyle='--', linewidth=3, alpha=0.8, 
              label='SOUND ONSET', zorder=11)
    
    # Add analysis windows as gray shading
    ax.axvspan(-analysis_duration, 0, alpha=0.15, color='gray', label=f'Pre Analysis ({analysis_duration}s)', zorder=0)
    ax.axvspan(0, analysis_duration, alpha=0.15, color='gray', label=f'Post Analysis ({analysis_duration}s)', zorder=0)
    
    # Add display window boundaries
    ax.axvline(x=-display_duration, color='lightgray', linestyle=':', alpha=0.7, label=f'Display Boundary (±{display_duration}s)')
    ax.axvline(x=display_duration, color='lightgray', linestyle=':', alpha=0.7)
    
    ax.set_xlabel('Time relative to sound (s)', fontsize=12)
    ax.set_ylabel('Δ F / F', fontsize=12)
    ax.set_title(f'All Subjects: Analysis ±{analysis_duration}s (Gray), Display ±{display_duration}s\n(n={len(individual_traces)} subjects)', 
                fontsize=14, fontweight='bold')
    
    # Set limits
    ax.set_xlim(-display_duration, display_duration)
    ax.set_ylim(global_ylim)
    
    # Position legend outside plot area
    if len(individual_traces) <= 10:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=4)
    else:
        # For many subjects, just show key elements
        handles = [ax.lines[-1], ax.collections[0] if ax.collections else None, 
                  [line for line in ax.lines if line.get_linestyle() == '--'][0]]
        labels = ['Group Average', 'Group SEM', 'Sound Onset']
        handles = [h for h in handles if h is not None]
        ax.legend(handles, labels[:len(handles)], bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Summary statistics display
    print(f"\n{'='*80}")
    print("CONTINUOUS TRACE ANALYSIS SUMMARY")
    print(f"{'='*80}")
    print(f"Data format: Combined pre+post in single columns")
    print(f"Pre duration: {params['pre_duration']}s, Post duration: {params['post_duration']}s")
    print(f"Sound onset at: {params['sound_onset_time']}s")
    print(f"ANALYSIS windows: ±{analysis_duration}s (gray shaded regions)")
    print(f"DISPLAY windows: ±{display_duration}s (full plot range)")
    print(f"  Analysis - Pre: {params['pre_window'][0]:.1f}s to {params['pre_window'][1]:.1f}s")
    print(f"  Analysis - Post: {params['post_window'][0]:.1f}s to {params['post_window'][1]:.1f}s")
    print(f"  Display - Pre: {params['pre_display_window'][0]:.1f}s to {params['pre_display_window'][1]:.1f}s")
    print(f"  Display - Post: {params['post_display_window'][0]:.1f}s to {params['post_display_window'][1]:.1f}s")
    print(f"Number of subjects with continuous traces: {len(individual_traces)}")
    print(f"All individual traces use same y-axis scale: {global_ylim[0]:.3f} to {global_ylim[1]:.3f}")
    
    # Calculate baseline adjustments summary
    baseline_shifts = [trace['baseline_shift'] for trace in individual_traces]
    print(f"Baseline adjustments applied:")
    print(f"  Mean shift: {np.mean(baseline_shifts):+.4f}")
    print(f"  Range: {np.min(baseline_shifts):+.4f} to {np.max(baseline_shifts):+.4f}")
    print(f"  Subjects requiring >0.01 adjustment: {sum(abs(shift) > 0.01 for shift in baseline_shifts)}")
    
    # Group trace info
    print(f"\nGroup average continuous trace:")
    print(f"  Display duration: {group_trace['combined_times'][-1] - group_trace['combined_times'][0]:.2f}s")
    print(f"  Total data points: {len(group_trace['combined_smooth'])}")
    print(f"  Smoothing window: ~{61*params['dt']:.2f}s")
    print(f"  Baseline adjustment: {-group_trace['baseline_shift']:+.4f}")
    print(f"\nSTATISTICAL ANALYSIS based on {analysis_duration}s windows only")

# Main execution
if __name__ == "__main__":
    # Replace with your actual file path
    csv_file_path = "/home/joeh/Documents/helicon_sound_pre_post.csv"
    
    # Data parameters - adjust these to match your data
    SAMPLING_RATE = 100.0  # Hz (100 Hz = 0.01s intervals)
    PRE_DURATION = 5.0     # seconds of pre-sound data
    POST_DURATION = 5.0    # seconds of post-sound data
    
    try:
        results = analyze_pre_post_data(
            csv_file_path, 
            sampling_rate=SAMPLING_RATE,
            pre_duration=PRE_DURATION,
            post_duration=POST_DURATION
        )
        
        print("\n" + "="*80)
        print("SUMMARY - ANALYSIS vs DISPLAY WINDOWS")
        print("="*80)
        
        data = results['averaged_analysis']
        params = results['analysis_params']
        tests = data['tests']
        analysis_duration = params['analysis_duration']
        display_duration = params['display_duration']
        
        print(f"Data format: {PRE_DURATION}s pre + {POST_DURATION}s post per column")
        print(f"Sound onset at: {params['sound_onset_time']:.1f}s")
        print(f"ANALYSIS windows (statistics): ±{analysis_duration}s (gray shaded)")
        print(f"DISPLAY windows (visualization): ±{display_duration}s (full range)")
        print(f"Number of subjects: {data['n_subjects']}")
        print(f"Valid subject columns: {data['valid_subjects']}")
        
        print(f"\nPre sound mean ({analysis_duration}s): {data['pre_stats']['mean']:.4f} ± {data['pre_stats']['std']:.4f}")
        print(f"Post sound mean ({analysis_duration}s): {data['post_stats']['mean']:.4f} ± {data['post_stats']['std']:.4f}")
        print(f"Mean change after sound: {data['post_stats']['mean'] - data['pre_stats']['mean']:.4f}")
        
        print(f"\nTime-series analysis (based on {analysis_duration}s windows):")
        print(f"  Independent t-test: p = {tests['independent_t'][1]:.6f}")
        if tests['paired_t'][1]:
            print(f"  Paired t-test: p = {tests['paired_t'][1]:.6f}")
        print(f"  Effect size (Cohen's d): {tests['cohens_d']:.4f}")
        
        print(f"\nSubject-level analysis (based on {analysis_duration}s windows):")
        print(f"  Paired t-test on subject means: p = {tests['subject_paired_t'][1]:.6f}")
        print(f"  Subject-level effect size: {tests['cohens_d_subjects']:.4f}")
        


        # Overall conclusion
        print(f"\n{'='*20} CONCLUSION {'='*20}")
        significant_tests = []
        if tests['independent_t'][1] < 0.05:
            significant_tests.append("time-series")
        if tests['subject_paired_t'][1] < 0.05:
            significant_tests.append("subject-level")
        
        if significant_tests:
            print(f"✓ SIGNIFICANT effects of sound found in: {', '.join(significant_tests)}")
        else:
            print("✗ No significant effects of sound found")
            
        # Effect size interpretation
        effect_size = abs(tests['cohens_d'])
        if effect_size > 0.8:
            print(f"✓ Large effect size detected (d = {effect_size:.3f})")
        elif effect_size > 0.5:
            print(f"~ Medium effect size detected (d = {effect_size:.3f})")
        elif effect_size > 0.2:
            print(f"~ Small effect size detected (d = {effect_size:.3f})")
        else:
            print(f"- Negligible effect size (d = {effect_size:.3f})")
            
        print(f"\nKey: Gray shaded regions in plots = Analysis windows ({analysis_duration}s)")
        print(f"      Full plot range = Display windows ({display_duration}s)")
        
        
    except FileNotFoundError:
        print(f"File not found: {csv_file_path}")
        print("Please check the file path and try again.")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please check your data format and try again.")
        import traceback
        traceback.print_exc()