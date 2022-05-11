def create_stack_bar_data(col, df):
    """
    Create a stacked bar plot of the data
    """
    aggregated = df[col].value_counts().sort_index()
    x_values = aggregated.index.tolist()
    y_values = aggregated.values.tolist()
    return x_values, y_values