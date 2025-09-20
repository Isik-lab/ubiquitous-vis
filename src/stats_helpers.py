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
    comparison_pairs=None  # optional: list of (cond1, cond2)
):
    from statsmodels.stats.multitest import multipletests, fdrcorrection
    from scipy.stats import wilcoxon
    from pymer4.models import Lmer

    pairs = []
    pvals = []

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
                    fit = model.fit().reset_index()
                    p = fit.loc[fit['index'].str.startswith(condition_var), 'P-val'].values[0]
                except Exception:
                    p = 1.0

            pairs.append(((mask, cond1), (mask, cond2)))
            pvals.append(p)
    if correction == 'bonferroni':
        _, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='bonferroni')
    elif correction == 'fdr':
        _, pvals_corrected = fdrcorrection(pvals, alpha=0.05, method='indep')
    else:
        raise ValueError(f"Unknown correction method: {correction}")
    return pairs, pvals_corrected
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
    parametric=True
):
    from scipy.stats import wilcoxon
    from statsmodels.stats.multitest import multipletests, fdrcorrection
    from pymer4.models import Lmer

    pairs = []
    pvals = []

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
                fit = model.fit()
                for feature_name, p in zip(fit.index, fit['P-val']):
                    if feature_name.startswith(condition_var):
                        cond = feature_name.replace(f"{condition_var}", "").strip()
                    else:
                        cond = feature_name.strip()
                    pairs.append(((mask, cond), (mask, cond)))
                    pvals.append(p)
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
    comparison_pairs=None
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
            comparison_pairs=comparison_pairs
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
            correction=correction
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
        region_pair = row['region_pair'].replace('_vs_', ' vs ').replace('left','').replace('right','').replace('both','')

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
        headers = ['Hemisphere', 'Region Pair', 'Df', 'Chi-Sq', 'p-value', 'Sig.']
    else:
        headers = ['Region Pair', 'Df', 'Chi-Sq', 'p-value', 'Sig.']
    table_latex = tabulate(rows, headers, tablefmt="latex_booktabs")

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
    
    if(split_hemi):
        hemis = ['left','right']
    else:
        hemis = ['both']

    for hemi in hemis:

        hemi_df = results_df[(results_df['axis'] == axis) &
                              (results_df['hemi1'] == hemi) &
                              (results_df['hemi2'] == hemi)].copy()

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

        valid_superregions = ['MT', 'STS', 'temporal', 'frontal']
        hemi_df = hemi_df[
            hemi_df['mask1_super'].isin(valid_superregions) &
            hemi_df['mask2_super'].isin(valid_superregions)
        ]

        hemi_df = hemi_df[hemi_df['mask1_super'] != hemi_df['mask2_super']]  # exclude same region pairs

        hemi_df['super_pair'] = hemi_df.apply(
            lambda row: '__'.join(sorted([row['mask1_super'], row['mask2_super']])), axis=1
        )

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

    combined_fits = pd.concat(all_fit_results, ignore_index=True)
    combined_contrasts = pd.concat(all_contrast_results, ignore_index=True)

    return combined_fits, combined_contrasts
def generate_latex_pairwise_similarity_table_with_hemisphere(
    results_df, 
    output_path,
    include_hemi=False
    ):
    from tabulate import tabulate

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

        left_label = f"{region_map[left_r1]}-{region_map[left_r2]}"
        right_label = f"{region_map[right_r1]}-{region_map[right_r2]}"
        return left_label, right_label

    rows = []

    for _, row in results_df.iterrows():
        hemi = row['hemisphere'].capitalize()

        region1, region2 = parse_regions(row['contrast'])
        beta = f"{row['estimate']:.3f}"
        pval_raw = row['p.value']
        
        if pval_raw < 0.001:
            pval = f"{pval_raw:.1e}"
        else:
            pval = f"{pval_raw:.3f}"

        # Significance stars
        if row['p.value'] < 0.001:
            stars = '***'
        elif row['p.value'] < 0.01:
            stars = '**'
        elif row['p.value'] < 0.05:
            stars = '*'
        else:
            stars = 'n.s.'

        if include_hemi:
            rows.append([hemi, region1, region2, beta, pval, stars])
        else:
            rows.append([region1, region2, beta, pval, stars])

    if include_hemi:
        headers = ['Hemisphere', 'Region Pair 1', 'Region Pair 2', 'Estimate', 'p-value', 'Sig.']
    else:
        headers = ['Region Pair 1', 'Region Pair 2', 'Estimate', 'p-value', 'Sig.']
    table_latex = tabulate(rows, headers, tablefmt="latex_booktabs")

    # Write LaTeX table to file
    with open(output_path, 'w') as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write(table_latex)
        f.write("\n\\caption{Pairwise region similarity comparisons.}\n\\end{table}")

    return table_latex