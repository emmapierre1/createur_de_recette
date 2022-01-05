DELETE_ROW_IF_FIRST_CHAR=[':','`', '¨', '➢']
REPLACE_IF_FIRST_CHAR = [' ', '-', '\n']
STR_TO_REPLACE_BY_SINGLE_OCCURENCES = ['!', '.', ' ', '-', ]



def recipe_df_cleaning(recipes_df) :
    """Function dedicated to the cleaning of the recipe's DataFrame cooking steps
    Dropped :
        - steps with HTML Tags in it '<'
        - steps starting with ':','`', '¨', '➢', '\n' ( only 1 encounters as first character in steps)

    Replaced characters :
        - if first char is ' ' or '-'

        - multiples !!! ... and spaces '    '

    """
    # Creating a column with the first letters of each recipe_steps to identify troublemakers
    recipes_df['first_letter'] = recipes_df.recipe_steps.apply(lambda x : x[0])

    # Deleting HTML Tags (balises)
    rows_to_drop = list(recipes_df.recipe_steps[recipes_df.recipe_steps.apply(lambda x : x.find('<')) != -1].index)


    recipes_df = drop_rows(recipes_df, rows_to_drop)

    recipes_df.drop(rows_to_drop, axis=0, inplace=True)

    # Deleting rows with first_letter in DELETE_ROW_IF_FIRST_CHAR
    for deleted_char in DELETE_ROW_IF_FIRST_CHAR :
        rows_to_drop = list(recipes_df.recipe_steps[recipes_df.recipe_steps.apply(lambda x : \
                                x.startswith(deleted_char))].index)

    recipes_df = drop_rows(recipes_df, rows_to_drop)
    # Replacing first chars of rows if it starts with a char from REPLACE_IF_FIRST_CHAR (Twice, just in cases '- ' or ' .')
    for i in range(2) :
        for replaced_char in REPLACE_IF_FIRST_CHAR :
            recipes_df.recipe_steps = recipes_df.recipe_steps.apply(lambda x : x[1:] \
                    if x.startswith(replaced_char) else x)

    # Replacing multiples occurences like !!!!! to ! or ... to . or '        ' to ' '
    for i in range(5) :
        for char in STR_TO_REPLACE_BY_SINGLE_OCCURENCES:
            recipes_df.recipe_steps = recipes_df.recipe_steps.apply(lambda x : x.replace(f"{char}{char}", f"{char}"))


    # dropping useless columns
    recipes_df.drop(columns='first_letter', inplace=True)



    return recipes_df

def drop_rows(df, rows_to_drop) :
    """ delete rows and reset index
    df : DataFrame
    rows_to_drop : list of indexes of rows to drop

    """
    df.drop(rows_to_drop, axis=0, inplace=True)
    df = df.reset_index().drop(columns='index')
    return df


#Description.
# Module permettant de réaliser les fonctionnalités suivantes :

#1. Supprimer les recettes qui ont (1 ) ou 0 ingrédients
#2. Supprimer recettes qui ont + de 20 ingrédients ?
#3. Supprimer les recettes qui n'ont pas d'étapes ou moins de 10 char
#4. Supprimer doublons
#5. Supprimer multiples espaces entre mots
#6. Etapes qui commencent avec '-' ou un char spécial ?
#7. Instructions avec balises html -> suppr 200 recettes
#8. Corriger ponctuation : !!!!, ?!, ...

#1. Supprimer les recettes qui ont (1 ) ou 0 ingrédients
#2. Supprimer recettes qui ont + de 20 ingrédients ?
def resample_ingredients(ingredients,min_ingredient=2, max_ingredient=10):
    #Return table ingredient with only min and max ingredients
    temp=ingredients.groupby("recipe_id").count()
    temp=temp.query(f"{min_ingredient}<=ingredient<={max_ingredient}")
    return ingredients[ingredients["recipe_id"].isin(temp.index)]
