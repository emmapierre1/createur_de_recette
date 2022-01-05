import pandas as pd

DELETE_ROW_IF_FIRST_CHAR=[':','`', '¨', '➢', "'", '[', '°', '_', '>', '.', '•', '"'] # '.' '_' and '•' pourraît être enlever pas une fonction qui supprime les puces
DROP_IF_SEQUENCE_IN_IT = ['http']

REPLACE_IF_FIRST_CHAR = [' ', '\n']
STR_TO_REPLACE_BY_SINGLE_OCCURENCES = ['!', '.', ' ', '-']
REPLACE_IF_SEQUENCE_IN_IT = [';-)', '~', '""']



#Description.
# Module permettant de réaliser les fonctionnalités suivantes :



#3. Supprimer les recettes qui n'ont pas d'étapes ou moins de 10 char


#6. Etapes qui commencent avec '-' ou un char spécial ?




#10 Travailler sur le char 1
# mettre toutes les premières lettres de chaque phrases en maj


"""
Cleared :
#1. Supprimer les recettes qui ont (1 ) ou 0 ingrédients
#2. Supprimer recettes qui ont + de 20 ingrédients ?

#4. Supprimer doublons

#5. Supprimer multiples espaces entre mots

#7. Instructions avec balises html -> suppr 200 recettes
#8. Corriger ponctuation : !!!!, ?!, ...
#9 Remplacez la supression d'espace par une méthode regex
"""
def ingredients_df_cleaning(ingredients_df) :

    ingredients_df.drop_duplicates(inplace=True)
    ingredients_df = resample_ingredients(ingredients_df,min_ingredient=2, max_ingredient=10)

    ingredients_df = ingredients_df.reset_index().drop(columns='index')
    return ingredients_df


def recipe_df_cleaning(recipes_df) :
    """Function dedicated to the cleaning of the recipe's DataFrame cooking steps
    Dropped :
        - steps with HTML Tags in it '<'
        - steps starting with ':','`', '¨', '➢', '\n' ( only 1 encounters as first character in steps)

    Replaced characters :
        - if first char is ' ' or '-'

        - multiples !!! ... and spaces '    '

    """
    recipes_df.drop_duplicates(inplace=True)
    recipes_df = recipes_df.reset_index().drop(columns='index')
    # Enlever les '- ' avec un chiffre derrière éventuellement
    recipes_df.recipe_steps = recipes_df.recipe_steps.replace(to_replace ='(^|\\n) *(\*|-|•|~)+ *\d* *', value = '\g<1>', regex = True)

    # Replace '(digit)' per an empty string
    recipes_df.recipe_steps = recipes_df.recipe_steps.replace(to_replace ='\(\d{1,2}\)', value = '', regex = True)

    # removing white space after each new line
    recipes_df = remove_whitespace_after_new_line(recipes_df)

    # transform sequences of multiples spaces to a single one
    recipes_df = recipes_df.replace(to_replace =' {2,}', value = ' ', regex = True)

    # Creating a column with the first letters of each recipe_steps to identify troublemakers
    recipes_df['first_letter'] = recipes_df.recipe_steps.apply(lambda x : x[0])

    # Deleting HTML Tags (balises)
    recipes_df = drop_rows_with_character_in_it(recipes_df, ['<'])

    # Deleting rows with characters in DROP_IF_SEQUENCE_IN_IT
    recipes_df = drop_rows_with_character_in_it(recipes_df)


    # Deleting rows with first_letter in DELETE_ROW_IF_FIRST_CHAR
    recipes_df = delete_row_if_first_char_in_it(recipes_df)

    # Replacing first chars of rows if it starts with a char from REPLACE_IF_FIRST_CHAR (Twice, just in cases '- ' or ' .')
    recipes_df = replace_first_char_of_string(recipes_df)

    # Replacing multiples occurences like !!!!! to ! , ... to . or '        ' to ' '
    recipes_df = replace_multiple_chars_by_a_single_one(recipes_df)

    # Replace sequences in REPLACE_IF_SEQUENCE_IN_IT by empty string or choosen string
    recipes_df = replace_sequences_in_column(recipes_df, REPLACE_IF_SEQUENCE_IN_IT, '')

    # Apply capitalize to each steps of the recipe_steps columns
    recipes_df = capitalize_steps(recipes_df)


    # dropping useless columns
    # recipes_df.drop(columns='first_letter', inplace=True)

    return recipes_df

def replace_first_char_of_string(df, chars_list=REPLACE_IF_FIRST_CHAR, iterations = 3) :
    """cut the first character of each recipe steps if it's in chars_list """
    for i in range(iterations) :
        for replaced_char in chars_list :
            df.recipe_steps = df.recipe_steps.apply(lambda x : x[1:] \
                    if x.startswith(replaced_char) else x)
    return df


def delete_row_if_first_char_in_it(df, chars_list=DELETE_ROW_IF_FIRST_CHAR) :
    "delete row if first char of recipe_steps is inside chars_list"

    for deleted_char in chars_list :
        rows_to_drop = list(df.recipe_steps[df.recipe_steps.apply(lambda x : \
                                x.startswith(deleted_char))].index)
        df = drop_rows(df, rows_to_drop)
    return df


def replace_multiple_chars_by_a_single_one(df, chars_list=STR_TO_REPLACE_BY_SINGLE_OCCURENCES, iteration=30) :
    "replace multiple occurences of a char like !!!!!! by !"
    for i in range(iteration):
         for char in chars_list:
            df.recipe_steps = df.recipe_steps.apply(lambda x : x.replace(f"{char}{char}", f"{char}"))
    return df

def replace_sequences_in_column(df, chars_list=REPLACE_IF_SEQUENCE_IN_IT, new_seq='') :
    for old_sequence in chars_list :
        df.recipe_steps = df.recipe_steps.apply(lambda x : x.replace(old_sequence, new_seq))
    return df

def drop_rows_with_character_in_it(recipes_df, sequence_list=DROP_IF_SEQUENCE_IN_IT):
    """ Drop recipe_steps with given character in it e.g. '<'
    char must be a string
    """
    for seq in sequence_list :
        rows_to_drop = list(recipes_df.recipe_steps[recipes_df.recipe_steps.apply(lambda x : x.find(seq)) != -1].index)
        recipes_df = drop_rows(recipes_df, rows_to_drop)
    return recipes_df

def drop_rows(df, rows_to_drop) :
    """ delete rows and reset index
    df : DataFrame
    rows_to_drop : list of indexes(int) of rows to drop

    """
    df.drop(rows_to_drop, axis=0, inplace=True)
    df = df.reset_index().drop(columns='index')
    return df

def remove_whitespace_after_new_line(df):
    #1. Remove whitespace before each new line
    df.recipe_steps = df.recipe_steps.apply(lambda x : x.replace("\n\n ","\n\n"))
    return df

def resample_ingredients(ingredients_df,min_ingredient=2, max_ingredient=10):
    #Return table ingredients_df with only min and max ingredients
    temp=ingredients_df.groupby("recipe_id").count()
    temp=temp.query(f"{min_ingredient}<=ingredient<={max_ingredient}")
    return ingredients_df[ingredients_df["recipe_id"].isin(temp.index)]

def capitalize_steps(df):
    """return the recipes's DataFrame with capitalized steps"""
    df.recipe_steps = df.recipe_steps.apply(lambda x : x.split("\n\n"))
    df.recipe_steps = df.recipe_steps.apply(capitalize_steps_loop)
    return df

def capitalize_steps_loop(steps_list):
    return "\n\n".join([step.capitalize() for step in steps_list])
