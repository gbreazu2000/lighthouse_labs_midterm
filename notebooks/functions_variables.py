def encode_tags(df):

    """Use this function to manually encode tags from each sale.
    You could also provide another argument to filter out low 
    counts of tags to keep cardinality to a minimum.
       
    Args:
        pandas.DataFrame

    Returns:
        pandas.DataFrame: modified with encoded tags
    """
    tags = df["tags"].tolist()
    # create a unique list of tags and then create a new column for each tag
        
    return df


#hyperparameter tuning search function
def hyperparameter_search(training_folds, validation_folds, param_grid, evaluation_metric):
    best_hyperparameters = None
    best_score = float('-inf')  # Initialize with a very low value

    # Generate all combinations of hyperparameters
    hyperparameter_combinations = list(product(*param_grid.values()))

    for hyperparams in hyperparameter_combinations:
        fold_scores = []
        
        # Create and train the model using the current hyperparameters
        for i in range(len(training_folds)):
            model = RandomForestRegressor(**dict(zip(param_grid.keys(), hyperparams)))
            model.fit(training_folds[i].drop(columns=['description.sold_price']), training_folds[i]['description.sold_price'])

            # Make predictions on the validation set
            predictions = model.predict(validation_folds[i].drop(columns=['description.sold_price']))
            score = evaluation_metric(validation_folds[i]['description.sold_price'], predictions)

            fold_scores.append(score)

        # Calculate the average score across all folds
        average_score = sum(fold_scores) / len(fold_scores)
        
        # Update best hyperparameters if the current combination is better
        if average_score > best_score:
            best_score = average_score
            best_hyperparameters = hyperparams

    return best_hyperparameters

#Manual cross-fold validation
def custom_cross_validation(training_data, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    training_folds = []
    validation_folds = []

    for train_index, val_index in kf.split(training_data):
        train_fold = training_data.iloc[train_index]
        val_fold = training_data.iloc[val_index]

        # Append folds to the respective lists
        training_folds.append(train_fold)
        validation_folds.append(val_fold)

    return training_folds, validation_folds