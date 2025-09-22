import pandas as pd

from src.data.normalization import robust_normalize

def sample_top_k_zipf_unique(df: pd.DataFrame, k: int = 1000):
    all_samples = []

    for pid, group in df.groupby('diag_exercise'):
        ast_hashes = group['code'].apply(robust_normalize)
        valid_group = group.loc[ast_hashes.index].copy()
        valid_group['ast_hash'] = ast_hashes.values
        valid_group = valid_group.dropna(subset=['ast_hash'])

        # Count and sort cluster frequencies
        cluster_counts = valid_group['ast_hash'].value_counts()
        total = len(group)
        sorted_hashes = cluster_counts.sort_values(ascending=False)

        # Select top-k hashes
        top_k_hashes = sorted_hashes.head(k).index

        # Sample one representative program per hash
        representatives = (
            valid_group[valid_group['ast_hash'].isin(top_k_hashes)]
            .groupby('ast_hash')
            .apply(lambda x: x.sample(n=1, random_state=42))
            .reset_index(drop=True)
        )

        # Add frequency info
        representatives['cluster_frequency'] = representatives['ast_hash'].map(cluster_counts)
        representatives['normalized_cluster_frequency'] = representatives['cluster_frequency'] / total
        representatives = representatives.sort_values(by="normalized_cluster_frequency", ascending=False)

        all_samples.append(representatives)

    return pd.concat(all_samples).reset_index(drop=True)


def sample_top_p_zipf_unique(df: pd.DataFrame, top_p=0.95):
    all_samples = []

    for pid, group in df.groupby('diag_exercise'):
        ast_hashes = group['code'].apply(robust_normalize)
        valid_group = group.loc[ast_hashes.index].copy()
        valid_group['ast_hash'] = ast_hashes.values
        valid_group = valid_group.dropna(subset=['ast_hash'])

        # Count and sort cluster frequencies
        cluster_counts = valid_group['ast_hash'].value_counts()
        total = len(group)
        sorted_hashes = cluster_counts.sort_values(ascending=False)
        cumulative = (sorted_hashes / total).cumsum()

        # Select hashes until cumulative frequency reaches top_p
        top_p_hashes = cumulative[cumulative <= top_p].index.tolist()
        if len(top_p_hashes) < len(cumulative):
            top_p_hashes.append(cumulative.index[len(top_p_hashes)])

        # Sample one representative program per selected hash
        selected_group = valid_group[valid_group['ast_hash'].isin(top_p_hashes)]
        representatives = (
            selected_group.groupby('ast_hash')
            .apply(lambda x: x.sample(n=1, random_state=42))
            .reset_index(drop=True)
        )

        # Add frequency info
        representatives['cluster_frequency'] = representatives['ast_hash'].map(cluster_counts)
        representatives['normalized_cluster_frequency'] = representatives['cluster_frequency'] / total
        representatives = representatives.sort_values(by="normalized_cluster_frequency", ascending=False)

        all_samples.append(representatives)

    return pd.concat(all_samples).reset_index(drop=True)



def sample_zipf(df, total=100, head=20, random_state=42):
    """
    For each diag_exercise:
        - Select the top `head` unique solutions by normalized frequency (head of the distribution).
        - Randomly sample (total - head) unique solutions from the remaining (tail).
        - Returns a dataframe with frequency info.
    """
    all_samples = []

    for pid, group in df.groupby('diag_exercise'):
        ast_hashes = group['code'].apply(robust_normalize)
        valid_group = group.loc[ast_hashes.index].copy()
        valid_group['ast_hash'] = ast_hashes.values
        valid_group = valid_group.dropna(subset=['ast_hash'])

        # Count cluster frequencies
        cluster_counts = valid_group['ast_hash'].value_counts()
        total_unique = len(cluster_counts)

        # Sort hashes by cluster frequency (head first)
        sorted_hashes = cluster_counts.sort_values(ascending=False)
        top_head_hashes = sorted_hashes.index[:head].tolist()
        tail_hashes = sorted_hashes.index[head:].tolist()

        # One representative per hash (program)
        head_representatives = (
            valid_group[valid_group['ast_hash'].isin(top_head_hashes)]
            .groupby('ast_hash')
            .apply(lambda x: x.sample(n=1, random_state=random_state))
            .reset_index(drop=True)
        )

        n_tail = max(0, total - len(head_representatives))
        # Sample from tail (excluding any hashes already in head)
        tail_group = valid_group[valid_group['ast_hash'].isin(tail_hashes)]
        if len(tail_group['ast_hash'].unique()) > 0 and n_tail > 0:
            sampled_tail_hashes = (
                pd.Series(tail_group['ast_hash'].unique())
                .sample(n=min(n_tail, len(tail_group['ast_hash'].unique())), random_state=random_state)
                .tolist()
            )
            tail_representatives = (
                tail_group[tail_group['ast_hash'].isin(sampled_tail_hashes)]
                .groupby('ast_hash')
                .apply(lambda x: x.sample(n=1, random_state=random_state))
                .reset_index(drop=True)
            )
        else:
            tail_representatives = pd.DataFrame(columns=valid_group.columns)

        reps = pd.concat([head_representatives, tail_representatives]).reset_index(drop=True)

        # Add cluster frequency info
        reps['cluster_frequency'] = reps['ast_hash'].map(cluster_counts)
        reps['normalized_cluster_frequency'] = reps['cluster_frequency'] / len(valid_group)
        reps = reps.sort_values(by="normalized_cluster_frequency", ascending=False)

        all_samples.append(reps)

    return pd.concat(all_samples).reset_index(drop=True)



def drop_duplicates(data):
    print("Dataframe before dropping duplicates")
    print(data.groupby("diag_exercise").diag_exercise.count())

    # Normalize code
    data["normalized_code"] = list(map(robust_normalize, data["code"]))

    # Drop rows where normalization failed
    # data = data.dropna(subset=["normalized_code"])

    # Compute raw frequencies per diag_exercise
    freq_df = (
        data.groupby(["diag_exercise", "normalized_code"])
        .size()
        .reset_index(name="cluster_frequency")
    )

    # Compute total submissions per exercise (before deduplication)
    total_counts = data.groupby("diag_exercise").size().to_dict()
    freq_df["normalized_cluster_frequency"] = freq_df.apply(
        lambda row: row["cluster_frequency"] / total_counts[row["diag_exercise"]],
        axis=1
    )

    # Merge frequency info back into main dataframe
    data = data.merge(freq_df, on=["diag_exercise", "normalized_code"], how="left")

    # Now drop duplicates
    data = data.drop_duplicates("normalized_code")

    print("Dataframe after dropping duplicates")
    print(data.groupby("diag_exercise").diag_exercise.count()) 
    return data 