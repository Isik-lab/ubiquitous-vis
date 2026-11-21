import pandas as pd

def compute_pairwise_stats(
    data,
    masks,
    condition_var,
    value_var,
    subject_var='subject',
    group_var='hemi_mask',
    method='lmer',  # or 'wilcoxon'
    correction='bonferroni',
    parametric=True,
    comparison_pairs=None,  # optional: list of (cond1, cond2)
    return_details=False  # also return a dataframe with each test's estimate, SE, df and t (LME only)
):
    from statsmodels.stats.multitest import multipletests, fdrcorrection
    from scipy.stats import wilcoxon
    from pymer4.models import Lmer

    pairs = []
    pvals = []
    details = []

    if comparison_pairs is None:
        all_conditions = data[condition_var].dropna().unique()
        comparison_pairs = [
            (c1, c2) for i, c1 in enumerate(all_conditions)
            for c2 in all_conditions[i+1:]
        ]   
    for mask in masks:
        for cond1, cond2 in comparison_pairs:
            subset = data[
                ((data[condition_var] == cond1) | (data[condition_var] == cond2)) &
                (data[group_var] == mask)
            ]

            if len(subset) < 2 or subset[condition_var].nunique() < 2:
                continue

            if method == 'wilcoxon' or not parametric:
                x = subset[subset[condition_var] == cond1][value_var].values
                y = subset[subset[condition_var] == cond2][value_var].values
                try:
                    stat = wilcoxon(x, y, alternative='two-sided', nan_policy='omit')
                    p = stat.pvalue
                except ValueError:
                    p = 1.0
            else:
                model = Lmer(f"{value_var} ~ 1 + {condition_var} + (1|{subject_var})", data=subset)
                try:
                    model.fit(summarize=False)
                    # model.coefs keeps full precision (the table fit() prints is rounded to 3 decimals)
                    fit = model.coefs.reset_index()
                    row = fit.loc[fit['index'].str.startswith(condition_var)].iloc[0]
                    p = row['P-val']
                    # the estimate is the non-reference level minus the reference level
                    level = row['index'][len(condition_var):]
                    reference = cond2 if level == cond1 else cond1
                    details.append({'test_index': len(pvals), 'mask': mask, 'condition': level, 'reference': reference,
                                    'n_subjects': subset.dropna(subset=[value_var])[subject_var].nunique(), 'estimate': row['Estimate'],
                                    'SE': row['SE'], 'df': row['DF'], 't': row['T-stat'], 'p': p})
                except Exception:
                    p = 1.0

            pairs.append(((mask, cond1), (mask, cond2)))
            pvals.append(p)

    # If no valid comparisons were made, return empty lists
    if len(pvals) == 0:
        return ([], [], pd.DataFrame()) if return_details else ([], [])

    if correction == 'bonferroni':
        _, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='bonferroni')
    elif correction == 'fdr':
        _, pvals_corrected = fdrcorrection(pvals, alpha=0.05, method='indep')
    else:
        raise ValueError(f"Unknown correction method: {correction}")
    if return_details:
        return pairs, pvals_corrected, _details_with_correction(details, pvals_corrected, correction)
    return pairs, pvals_corrected
def _details_with_correction(details, pvals_corrected, correction):
    """Add the corrected p-values and matching 95% CIs to the per-test LME details (one row per successful fit;
    test_index is the test's position in pvals_corrected). For Bonferroni, the CIs are adjusted over the same
    family as the p-values, so a CI excludes zero exactly when the corrected p-value is below 0.05."""
    from scipy import stats
    details = pd.DataFrame(details)
    if details.empty:
        return details
    n_tests = len(pvals_corrected)
    details['p_corrected'] = [pvals_corrected[i] for i in details['test_index']]
    alpha = 0.05 / n_tests if correction == 'bonferroni' else 0.05
    t_crit = stats.t.ppf(1 - alpha / 2, details['df'])
    details['ci_low'] = details['estimate'] - t_crit * details['SE']
    details['ci_high'] = details['estimate'] + t_crit * details['SE']
    details['correction'] = correction
    details['n_tests'] = n_tests
    return details.drop(columns='test_index')
def compare_against_zero(
    data,
    masks,
    condition_levels,
    condition_var,
    value_var,
    subject_var='subject',
    group_var='hemi_mask',
    method='lmer',
    correction='bonferroni',
    parametric=True,
    return_details=False  # also return a dataframe with each test's estimate, SE, df and t (LME only)
):
    from scipy.stats import wilcoxon
    from statsmodels.stats.multitest import multipletests, fdrcorrection
    from pymer4.models import Lmer

    pairs = []
    pvals = []
    details = []

    for mask in masks:
        subset = data[data[group_var] == mask]

        if subset.empty:
            continue

        if method == 'wilcoxon' or not parametric:
            for cond in condition_levels:
                cond_data = subset[subset[condition_var] == cond]
                x = cond_data[value_var].values
                try:
                    stat = wilcoxon(x, alternative='greater', nan_policy='omit')
                    p = stat.pvalue
                except ValueError:
                    p = 1.0
                pairs.append(((mask, cond), (mask, cond)))
                pvals.append(p)
        else:
            try:
                model = Lmer(
                    f"{value_var} ~ 0 + {condition_var} + (1|{subject_var})",
                    data=subset
                )
                model.fit(summarize=False)
                fit = model.coefs  # full precision (the table fit() prints is rounded to 3 decimals)
                for feature_name, p in zip(fit.index, fit['P-val']):
                    if feature_name.startswith(condition_var):
                        cond = feature_name.replace(f"{condition_var}", "").strip()
                    else:
                        cond = feature_name.strip()
                    pairs.append(((mask, cond), (mask, cond)))
                    pvals.append(p)
                    row = fit.loc[feature_name]
                    details.append({'test_index': len(pvals) - 1, 'mask': mask, 'condition': cond,
                                    'n_subjects': subset.dropna(subset=[value_var])[subject_var].nunique(), 'estimate': row['Estimate'],
                                    'SE': row['SE'], 'df': row['DF'], 't': row['T-stat'], 'p': p})
            except Exception as e:
                print(f"Error fitting model for {mask}: {e}")
                for cond in condition_levels:
                    pairs.append(((mask, cond), (mask, cond)))
                    pvals.append(1.0)

    if correction == 'bonferroni':
        _, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='bonferroni')
    elif correction == 'fdr':
        _, pvals_corrected = fdrcorrection(pvals, alpha=0.05, method='indep')
    else:
        raise ValueError(f"Unknown correction method: {correction}")

    if return_details:
        return pairs, pvals_corrected, _details_with_correction(details, pvals_corrected, correction)
    return pairs, pvals_corrected
def compare_selectivity_between_SI_and_language(
    data,
    condition, #condition to subtract from
    subtract_condition, #condition to subtract
    roi_column='hemi_mask',
    subject_column='subject',
    value_column='glm_weight'
):
    import pandas as pd
    from pymer4.models import Lmer

    subset = data[data['glm_response_contrast'].isin([condition, subtract_condition])]
    pivoted = subset.pivot_table(index=[subject_column, roi_column], columns='glm_response_contrast', values=value_column).reset_index()
    pivoted['selectivity'] = pivoted[condition] - pivoted[subtract_condition]

    #this collapses across posterior and anterior and left and right
    def categorize_roi(hemi_mask):
        if 'STS' in hemi_mask:
            return 'social'
        elif ('temporal' in hemi_mask) | ('frontal' in hemi_mask) | ('Temp' in hemi_mask):
            return 'language'
        else:
            return 'MT'

    pivoted['roi_type'] = pivoted[roi_column].apply(categorize_roi)
    pivoted = pivoted[pivoted['roi_type']!='MT'] #only include social interaction and language regions
    model = Lmer("selectivity ~ roi_type + (1|subject)", data=pivoted.rename(columns={subject_column: 'subject'}))
    fit = model.fit()
    return fit
def compare_selectivity_within_region_type_per_hemisphere(
    data,
    condition,  # condition to subtract from
    subtract_condition,  # condition to subtract
    roi_column='hemi_mask',
    subject_column='subject',
    value_column='glm_weight'
):
    import pandas as pd
    from pymer4.models import Lmer

    subset = data[data['glm_response_contrast'].isin([condition, subtract_condition])]
    pivoted = subset.pivot_table(
        index=[subject_column, roi_column],
        columns='glm_response_contrast',
        values=value_column
    ).reset_index()

    pivoted['selectivity'] = pivoted[condition] - pivoted[subtract_condition]

    # Parse hemisphere from hemi_mask
    def parse_hemisphere(hemi_mask):
        if hemi_mask.startswith('left'):
            return 'left'
        elif hemi_mask.startswith('right'):
            return 'right'
        else:
            return 'unknown'

    def parse_roi_type(hemi_mask):
        if 'STS' in hemi_mask:
            return 'social'
        elif ('temporal' in hemi_mask) or ('frontal' in hemi_mask) or ('Temp' in hemi_mask):
            return 'language'
        else:
            return 'MT'

    def parse_subregion(hemi_mask):
        if ('pSTS' in hemi_mask) or ('pTemp' in hemi_mask):
            return 'posterior'
        elif ('aSTS' in hemi_mask) or ('aTemp' in hemi_mask):
            return 'anterior'
        else:
            return 'other'

    pivoted['hemisphere'] = pivoted[roi_column].apply(parse_hemisphere)
    pivoted['roi_type'] = pivoted[roi_column].apply(parse_roi_type)
    pivoted['subregion'] = pivoted[roi_column].apply(parse_subregion)

    # Only keep social and language ROIs (ignore MT and others)
    pivoted = pivoted[pivoted['roi_type'].isin(['social', 'language'])]
    pivoted = pivoted[pivoted['subregion'].isin(['posterior', 'anterior'])]

    all_fits = []

    for region_type in ['social', 'language']:
        for hemisphere in ['left', 'right']:
            region_data = pivoted[
                (pivoted['roi_type'] == region_type) &
                (pivoted['hemisphere'] == hemisphere)
            ].copy()

            if len(region_data) >= 3:  # minimum data to avoid singular fit
                model = Lmer("selectivity ~ subregion + (1|subject)", 
                             data=region_data.rename(columns={subject_column: 'subject'}))
                fit = model.fit().reset_index()
                fit['region_type'] = region_type
                fit['hemisphere'] = hemisphere
                all_fits.append(fit)

    combined_fit_df = pd.concat(all_fits, ignore_index=True)

    return combined_fit_df
def compare_selectivity_within_region_type(
    data,
    condition,  # condition to subtract from
    subtract_condition,  # condition to subtract
    roi_column='hemi_mask',
    subject_column='subject',
    value_column='glm_weight'
):
    import pandas as pd
    from pymer4.models import Lmer

    subset = data[data['glm_response_contrast'].isin([condition, subtract_condition])]
    pivoted = subset.pivot_table(
        index=[subject_column, roi_column],
        columns='glm_response_contrast',
        values=value_column
    ).reset_index()

    pivoted['selectivity'] = pivoted[condition] - pivoted[subtract_condition]

    # Assign both roi_type (social vs language) and subregion (posterior vs anterior)
    def parse_roi_type(hemi_mask):
        if 'STS' in hemi_mask:
            return 'social'
        elif ('temporal' in hemi_mask) or ('frontal' in hemi_mask) or ('Temp' in hemi_mask):
            return 'language'
        else:
            return 'MT'

    def parse_subregion(hemi_mask):
        if ('pSTS' in hemi_mask) or ('pTemp' in hemi_mask):
            return 'posterior'
        elif ('aSTS' in hemi_mask) or ('aTemp' in hemi_mask):
            return 'anterior'
        else:
            return 'other'

    pivoted['roi_type'] = pivoted[roi_column].apply(parse_roi_type)
    pivoted['subregion'] = pivoted[roi_column].apply(parse_subregion)

    # Only keep social and language ROIs (ignore MT and others)
    pivoted = pivoted[pivoted['roi_type'].isin(['social', 'language'])]

    all_fits = []

    for region_type in ['social', 'language']:
        region_data = pivoted[pivoted['roi_type'] == region_type].copy()
        model = Lmer("selectivity ~ subregion + (1|subject)", data=region_data.rename(columns={subject_column: 'subject'}))
        fit = model.fit().reset_index()
        fit['region_type'] = region_type
        all_fits.append(fit)
    
    # Merge everything into one dataframe
    combined_fit_df = pd.concat(all_fits, ignore_index=True)

    return combined_fit_df
def run_statistical_comparisons(
    data,
    masks,
    mode,
    condition_levels,
    condition_var,
    value_var,
    subject_var='subject',
    group_var='hemi_mask',
    method='lmer',
    parametric=True,
    correction='bonferroni',
    comparison_pairs=None,
    return_details=False
):
    if mode == 'compare_features':
        return compute_pairwise_stats(
            data=data,
            masks=masks,
            condition_var=condition_var,
            value_var=value_var,
            subject_var=subject_var,
            group_var=group_var,
            method=method,
            parametric=parametric,
            comparison_pairs=comparison_pairs,
            return_details=return_details
        )
    elif mode == 'compare_to_zero':
        return compare_against_zero(
            data=data,
            masks=masks,
            condition_levels=condition_levels,
            condition_var=condition_var,
            value_var=value_var,
            subject_var=subject_var,
            group_var=group_var,
            method=method,
            parametric=parametric,
            correction=correction,
            return_details=return_details
        )
    elif mode == 'compare_selectivity':
        return compare_selectivity_between_rois(data)
    elif mode == 'compare_hemispheric':
        return compare_hemispheric_distribution(data)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
def run_anova_per_hemisphere(
    data,
    plot_features,
):
    from pymer4.models import Lmer
    from rpy2.robjects import r
    from rpy2.robjects.packages import importr
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()
    all_anova_results = []

    hemispheres = data['hemisphere'].unique()
    regions = set(data['mask'])
    for hemi in hemispheres:
        print(f'Running ANOVA for {hemi} hemisphere: testing response profiles across regions')
        
        # Subset and filter
        hemi_data = data[(data['hemisphere'] == hemi) & (data['Feature_Space'].isin(plot_features))].copy()
        print(hemi_data)
        # Fit models
        model1 = Lmer("encoding_response ~ 1 + Feature_Space + hemi_mask + (1|subject)", data=hemi_data)
        model1.fit()

        model2 = Lmer("encoding_response ~ 1 + Feature_Space*hemi_mask + (1|subject)", data=hemi_data)
        model2.fit()

        # Run ANOVA
        anova_result = r["anova"](model1.model_obj, model2.model_obj)
        anova_df = pandas2ri.rpy2py(anova_result)
        anova_df["hemisphere"] = hemi  # tag with hemisphere for clarity

        all_anova_results.append(anova_df)

    # Return full concatenated result
    return pd.concat(all_anova_results, ignore_index=True)
def run_anova_per_region_pair_per_hemisphere(
    data,
    plot_features,
    variable='Feature_Space'
):
    from itertools import combinations
    from pymer4.models import Lmer
    from rpy2.robjects import r
    from rpy2.robjects.packages import importr
    from rpy2.robjects import pandas2ri
    from statsmodels.stats.multitest import multipletests
    import pandas as pd

    pandas2ri.activate()
    all_anova_results = []

    hemispheres = data['hemisphere'].unique()
    data['hemi_mask'] = [hemi+'_'+ mask for hemi,mask in zip(data['hemisphere'],data['mask'])]

    for hemi in hemispheres:
        print(f'Running ANOVA for {hemi} hemisphere per region pair')

        hemi_data = data[(data['hemisphere'] == hemi) & (data[variable].isin(plot_features))].copy()

        # Get all pairs of regions
        regions = hemi_data['hemi_mask'].unique()
        region_pairs = list(combinations(regions, 2))

        hemi_results = []

        for region_pair in region_pairs:
            reg1, reg2 = region_pair
            print(f'  Testing pair: {reg1} vs {reg2}')
            
            # Subset to just these two regions
            pair_data = hemi_data[hemi_data['hemi_mask'].isin([reg1, reg2])].copy()

            try:
                model1 = Lmer(f"encoding_response ~ 1 + {variable} + hemi_mask + (1|subject)", data=pair_data)
                model1.fit()

                model2 = Lmer(f"encoding_response ~ 1 + {variable}*hemi_mask + (1|subject)", data=pair_data)
                model2.fit()

                anova_result = r["anova"](model1.model_obj, model2.model_obj)
                anova_df = pandas2ri.rpy2py(anova_result)
                anova_df["hemisphere"] = hemi
                anova_df["region_pair"] = f"{reg1}_vs_{reg2}"

                hemi_results.append(anova_df)

            except Exception as e:
                print(f"Failed for {hemi}, {reg1} vs {reg2}: {e}")
                continue

        if hemi_results:
            hemi_results_df = pd.concat(hemi_results, ignore_index=True)

            # Apply Bonferroni correction to interaction p-values
            pvals = hemi_results_df["Pr(>Chisq)"].values
            _, pvals_bonf, _, _ = multipletests(pvals, method='bonferroni')
            hemi_results_df["pval_bonferroni"] = pvals_bonf

            all_anova_results.append(hemi_results_df)

    # Return full concatenated result
    return pd.concat(all_anova_results, ignore_index=True) if all_anova_results else pd.DataFrame()
def generate_latex_anova_pairwise_table(anova_df, output_path,include_hemi=False):
    from tabulate import tabulate
    import numpy as np

    rows = []

    if include_hemi:
        # We only want the second row (interaction test row) per region_pair:
        interaction_rows = anova_df.groupby(['hemisphere', 'region_pair']).nth(1).reset_index()
    else:
        interaction_rows = anova_df.groupby(['region_pair']).nth(1).reset_index()

    for _, row in interaction_rows.iterrows():
        region_pair = row['region_pair']
        for hemi_prefix in ['left_', 'right_', 'both_']:
            region_pair = region_pair.replace(hemi_prefix, '')
        region_pair = region_pair.replace('_vs_', ' vs ')

        chisq = f"{row['Chisq']:.3f}"
        chi_df = int(row['Df'])
        pval_raw = row['pval_bonferroni']

        # Format p-value
        if pval_raw < 0.001:
            pval = f"{pval_raw:.1e}"
        else:
            pval = f"{pval_raw:.3f}"

        # Significance stars
        if pval_raw < 0.001:
            stars = '***'
        elif pval_raw < 0.01:
            stars = '**'
        elif pval_raw < 0.05:
            stars = '*'
        else:
            stars = 'n.s.'

        if include_hemi:
            hemi = row['hemisphere'].capitalize()
            rows.append([hemi, region_pair, chi_df, chisq, pval, stars])
        else:
            rows.append([region_pair, chi_df, chisq, pval, stars])
            
    if include_hemi:
        headers = ['Hemisphere', 'Region Pair', 'df', 'Chi-Sq', 'p-value', 'Sig.']
    else:
        headers = ['Region Pair', 'df', 'Chi-Sq', 'p-value', 'Sig.']
    colalign = (['left'] if include_hemi else []) + ['left', 'right', 'right', 'right', 'left']
    # keep the formatted strings as they are (tabulate would otherwise reparse them as numbers)
    table_latex = tabulate(rows, headers, tablefmt="latex_booktabs", disable_numparse=True, colalign=colalign)

    # Write LaTeX file
    with open(output_path, 'w') as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write(table_latex)
        f.write("\n\\caption{Pairwise ANOVA comparisons.}\n\\end{table}")

    return table_latex
def run_similarity_superregion_LME(results_df, axis, split_hemi=True,average_posterior_anterior=False):
    import pandas as pd
    import numpy as np
    from pymer4.models import Lmer
    from rpy2.robjects import r
    from rpy2.robjects import pandas2ri
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr

    emmeans = importr('emmeans')
    pandas2ri.activate()

    def map_to_superregion(label):
        parcel = label.split('-')[-1]
        if parcel in ['pSTS', 'aSTS','STS']:
            return 'STS'
        elif parcel in ['pTemp', 'aTemp', 'temporal']:
            return 'temporal'
        elif parcel == 'frontal':
            return 'frontal'
        elif parcel == 'MT':
            return 'MT'
        else:
            return 'other'

    all_fit_results = []
    all_contrast_results = []

    print(f"[run_similarity_superregion_LME] called with axis={axis!r}, split_hemi={split_hemi}, average_posterior_anterior={average_posterior_anterior}")
    print(f"  results_df shape={results_df.shape}, columns={list(results_df.columns)}")
    if 'axis' in results_df.columns:
        print(f"  unique axes in results_df: {results_df['axis'].unique().tolist()}")
    if 'hemi1' in results_df.columns:
        print(f"  unique hemi1 values: {results_df['hemi1'].unique().tolist()}")
    if 'hemi2' in results_df.columns:
        print(f"  unique hemi2 values: {results_df['hemi2'].unique().tolist()}")

    if(split_hemi):
        hemis = ['left','right']
    else:
        hemis = ['both']

    for hemi in hemis:

        axis_mask = (results_df['axis'] == axis)
        hemi1_mask = (results_df['hemi1'] == hemi)
        hemi2_mask = (results_df['hemi2'] == hemi)
        print(f"  [hemi={hemi}] axis match rows={axis_mask.sum()}, hemi1 match={hemi1_mask.sum()}, hemi2 match={hemi2_mask.sum()}, all three={(axis_mask & hemi1_mask & hemi2_mask).sum()}")

        hemi_df = results_df[axis_mask & hemi1_mask & hemi2_mask].copy()
        print(f"  [hemi={hemi}] hemi_df after axis+hemi filter: shape={hemi_df.shape}")

        # collapse posterior-anterior if needed (language only)
        def collapse_label(label):
            parts = label.split('-')
            name = '-'.join(parts[:-1])
            roi = parts[-1]
            if average_posterior_anterior and roi in ['pTemp', 'aTemp']:
                roi = 'temporal'
            return f"{name}-{roi}"

        hemi_df['mask1_name'] = hemi_df['mask1_name'].apply(collapse_label)
        hemi_df['mask2_name'] = hemi_df['mask2_name'].apply(collapse_label)

        # map superregions
        hemi_df['mask1_super'] = hemi_df['mask1_name'].apply(map_to_superregion)
        hemi_df['mask2_super'] = hemi_df['mask2_name'].apply(map_to_superregion)
        if not hemi_df.empty:
            print(f"  [hemi={hemi}] mask1_super counts: {hemi_df['mask1_super'].value_counts().to_dict()}")
            print(f"  [hemi={hemi}] mask2_super counts: {hemi_df['mask2_super'].value_counts().to_dict()}")
            print(f"  [hemi={hemi}] sample mask1_name -> mask1_super: {list(zip(hemi_df['mask1_name'].head(5), hemi_df['mask1_super'].head(5)))}")

        valid_superregions = ['MT', 'STS', 'temporal', 'frontal']
        hemi_df = hemi_df[
            hemi_df['mask1_super'].isin(valid_superregions) &
            hemi_df['mask2_super'].isin(valid_superregions)
        ]
        print(f"  [hemi={hemi}] after valid_superregions filter: shape={hemi_df.shape}")

        hemi_df = hemi_df[hemi_df['mask1_super'] != hemi_df['mask2_super']]  # exclude same region pairs
        print(f"  [hemi={hemi}] after excluding same-region pairs: shape={hemi_df.shape}")

        if hemi_df.empty:
            print(f"  [run_similarity_superregion_LME] no rows for hemi='{hemi}' axis='{axis}' — skipping (CSV likely generated with different split_hemi)")
            continue

        hemi_df['super_pair'] = hemi_df.apply(
            lambda row: '__'.join(sorted([row['mask1_super'], row['mask2_super']])), axis=1
        )
        # one value per participant per region pair (the sub-region pairs, e.g. pSTS-MT and aSTS-MT, are averaged),
        # as plotted in Figure 4; otherwise a participant contributes 1-4 non-independent rows per region pair
        hemi_df = hemi_df.groupby(['subject', 'super_pair'], as_index=False)['corr'].mean()

        # Fit model on super_pair
        model = Lmer("corr ~ super_pair + (1|subject)", data=hemi_df)
        fit = model.fit()

        fit_df = fit.reset_index()
        fit_df['hemisphere'] = hemi
        all_fit_results.append(fit_df)

        emmeans_obj = emmeans.emmeans(model.model_obj, "super_pair")
        contrast_obj = emmeans.contrast(emmeans_obj, method="pairwise", adjust="bonferroni")

        contrast_df_r = robjects.r['as.data.frame'](contrast_obj)
        contrast_df = pandas2ri.rpy2py(contrast_df_r)
        contrast_df['hemisphere'] = hemi
        all_contrast_results.append(contrast_df)

    print(f"[run_similarity_superregion_LME] collected {len(all_fit_results)} fit result(s) and {len(all_contrast_results)} contrast result(s) across hemis={hemis}")
    if not all_fit_results or not all_contrast_results:
        raise ValueError(
            f"run_similarity_superregion_LME: no hemispheres yielded data for axis={axis!r}, "
            f"split_hemi={split_hemi}, average_posterior_anterior={average_posterior_anterior}. "
            f"See printed diagnostics above."
        )

    combined_fits = pd.concat(all_fit_results, ignore_index=True)
    combined_contrasts = pd.concat(all_contrast_results, ignore_index=True)

    return combined_fits, combined_contrasts
def generate_latex_lme_details_table(details_df, output_path, mode='compare_to_zero', features=None, first=None):
    """LaTeX table of the per-test LME results returned by run_statistical_comparisons(..., return_details=True),
    one row per region (hemisphere) and feature: estimate, Bonferroni-adjusted 95% CI, SE, t (df), corrected p.
    mode='compare_to_zero' tests each feature against zero; mode='compare_features' is the difference between two
    features (condition minus reference).
    features: only list these features, in this order (the p-values and CIs stay corrected over all the tests).
    first: for compare_features, features to put first in the contrast (the sign is flipped when one is the reference)."""
    if mode == 'compare_features' and first is not None:
        flip = details_df['reference'].isin(first)
        details_df = details_df.copy()
        details_df.loc[flip, ['condition', 'reference']] = details_df.loc[flip, ['reference', 'condition']].values
        details_df.loc[flip, ['estimate', 't']] = -details_df.loc[flip, ['estimate', 't']]
        details_df.loc[flip, ['ci_low', 'ci_high']] = -details_df.loc[flip, ['ci_high', 'ci_low']].values
    feature_labels = {'Image Model': 'Image', 'Motion Model': 'Motion', 'Speech Model': 'Speech',
                      'word2vec_chunk120': 'Word', 'Sentence Model': 'Sentence',
                      'GPT2_chunk120': 'GPT-2', 'hubert_chunk120': 'HuBERT'}
    if features is not None:
        details_df = details_df[details_df['condition'].isin(features)]
        mask_order = {m: i for i, m in enumerate(details_df['mask'].unique())}
        details_df = details_df.assign(_mask_order=details_df['mask'].map(mask_order),
                                       _feature_order=details_df['condition'].map({f: i for i, f in enumerate(features)}))
        details_df = details_df.sort_values(['_mask_order', '_feature_order'])

    def num(x, fmt):
        # proper minus sign in LaTeX
        return f"{x:{fmt}}".replace('-', '$-$')

    def pval(p):
        return '$<$0.001' if p < 0.001 else f"{p:.3f}"

    def stars(p):
        return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'

    rows = []
    for _, row in details_df.iterrows():
        hemi, region = row['mask'].split(' ', 1)
        feature = feature_labels.get(row['condition'], row['condition'])
        if mode == 'compare_features':
            feature = f"{feature} $-$ {feature_labels.get(row['reference'], row['reference'])}"
        rows.append([region, hemi.capitalize(), feature, str(int(row['n_subjects'])), num(row['estimate'], '.4f'),
                     f"[{num(row['ci_low'], '.4f')}, {num(row['ci_high'], '.4f')}]", f"{row['SE']:.4f}",
                     f"{num(row['t'], '.2f')} ({row['df']:.1f})", pval(row['p_corrected']), stars(row['p_corrected'])])

    headers = ['Region', 'Hemisphere', 'Feature' if mode == 'compare_to_zero' else 'Contrast', '$n$', 'Estimate',
               '95\\% CI', 'SE', '$t$ (df)', '$p$', 'Sig.']
    # same layout as the other supplementary tables (table + tabular with booktabs rules)
    lines = ["\\begin{tabular}{lllrrcrrrl}", "\\toprule", ' & '.join(headers) + ' \\\\', "\\midrule"]
    lines += [' & '.join(r) + ' \\\\' for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    table_latex = '\n'.join(lines)
    with open(output_path, 'w') as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write(table_latex)
        f.write("\n\\caption{Linear mixed-effects model tests.}\n\\end{table}\n")
    return table_latex
def generate_latex_pairwise_similarity_table_with_hemisphere(
    results_df, 
    output_path,
    include_hemi=False
    ):
    from scipy import stats

    # Region label mapping
    region_map = {
        'MT': 'MT',
        'STS': 'SI',
        'temporal': 'tempLang',
        'frontal': 'frontLang',
    }

    # Function to parse and format region pairs
    def parse_regions(contrast_str):
        # contrast looks like: 'MT__STS - MT__temporal'
        left, right = contrast_str.split(' - ')
        left_r1, left_r2 = left.split('__')
        right_r1, right_r2 = right.split('__')

        left_label = f"{region_map[left_r1]}--{region_map[left_r2]}"
        right_label = f"{region_map[right_r1]}--{region_map[right_r2]}"
        return left_label, right_label

    def num(x, fmt):
        # proper minus sign in LaTeX
        return f"{x:{fmt}}".replace('-', '$-$')

    rows = []

    for hemi_name, hemi_df in results_df.groupby('hemisphere', sort=False):
        hemi = hemi_name.capitalize()
        # 95% CIs Bonferroni adjusted over the same family as the p-values (all contrasts in this hemisphere),
        # so a CI excludes zero exactly when the adjusted p-value is below 0.05
        n_contrasts = len(hemi_df)
        for _, row in hemi_df.iterrows():
            region1, region2 = parse_regions(row['contrast'])
            t_crit = stats.t.ppf(1 - 0.05 / (2 * n_contrasts), row['df'])
            estimate = num(row['estimate'], '.3f')
            ci = f"[{num(row['estimate'] - t_crit * row['SE'], '.3f')}, {num(row['estimate'] + t_crit * row['SE'], '.3f')}]"
            se = f"{row['SE']:.3f}"
            t_df = f"{num(row['t.ratio'], '.2f')} ({row['df']:.1f})"
            pval_raw = row['p.value']

            if pval_raw < 0.001:
                mantissa, exponent = f"{pval_raw:.1e}".split('e')
                pval = f"${mantissa}\\times10^{{{int(exponent)}}}$"
            else:
                pval = f"{pval_raw:.3f}"

            # Significance stars
            if pval_raw < 0.001:
                stars = '***'
            elif pval_raw < 0.01:
                stars = '**'
            elif pval_raw < 0.05:
                stars = '*'
            else:
                stars = 'n.s.'

            values = [region1, region2, estimate, ci, se, t_df, pval, stars]
            rows.append([hemi] + values if include_hemi else values)

    # one row per pair of region pairs; written by hand (not tabulate) so the LaTeX math is kept
    headers = ['Region pair 1', 'Region pair 2', 'Estimate', '95\\% CI', 'SE', '$t$ (df)', '$p$', 'Sig.']
    colspec = 'llrcrrrl'
    if include_hemi:
        headers = ['Hemisphere'] + headers
        colspec = 'l' + colspec
    lines = [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
             ' & '.join(headers) + ' \\\\', "\\midrule"]
    lines += [' & '.join(r) + ' \\\\' for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    table_latex = '\n'.join(lines)

    # Write LaTeX table to file (replace the whole table, including the column spec: it has 8 columns)
    with open(output_path, 'w') as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write(table_latex)
        f.write("\n\\caption{Pairwise region similarity comparisons.}\n\\end{table}\n")

    return table_latex