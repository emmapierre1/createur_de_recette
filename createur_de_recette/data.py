import pandas as pd

def load_data(nrows=None):
    """
    Loads the recipes and ingredients dataframes from CSV files.
    """
    return (pd.read_csv("data/recipes.csv", nrows=nrows), pd.read_csv("data/ingredients.csv"))

def recipe_to_string(ingredients_list, instructions):
    ingredients_string = '\n'.join(ingredients_list)
    return '🥕\n\n' + ingredients_string + '\n\n📝\n\n' + instructions

def get_recipes_string_list(nrows=None):
    """
    Returns a numpy array with the ingredients and instructions concatenated in a string.
    recipe_df is the recipe dataframe or subsample.
    ingredients_df can be the full ingredients dataset. It will be filtered according to recipe_df.
    """
    recipes_df, ingredients_df = load_data(nrows)
    ingredients_list_df = pd.DataFrame(ingredients_df.groupby('recipe_id')['ingredient'].apply(list)).reset_index().rename(columns={'ingredient':'ingredients_list'})
    return recipes_df.merge(ingredients_list_df, how='inner').apply(lambda x: recipe_to_string(x['ingredients_list'], x['recipe_steps']), axis=1)


def clean_data(df):
    """
    Return a preprocessed df
    Should we create those methods for each DataFrame ?
    """
    pass

if __name__ == '__main__':
    pass
