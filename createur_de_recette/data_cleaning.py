import pandas as pd
import re

DELETE_ROW_IF_FIRST_CHAR=[':','`', '¨', '➢', "'", '[', '°', '_', '>', '.', '•', '"'] # '.' '_' and '•' pourraît être enlever pas une fonction qui supprime les puces
DROP_IF_SEQUENCE_IN_IT = ['http', '\t', ':p', '\n:', ':o', ':D', ':-', 'Oo', 'A/', 'A)', '.!', '.?', '?.', '!.', 'I)', '. . .', '2ème étape', '1ère étape', 'étape 1', ',\n\n'] # check regex for \n\n2 , '\n\n2'
INDIVIDUALS_RECIPE_ID_TO_DROP = [55088, 14288, 168384, 382724, 383586, 11034, 24993, 323345, 28567, 165862, 25067, 94668, 48295, 52925] #  '$', '`', '*' added those to drop above to optimize model's speed , '$', '`', '*', '}', '§', '?', '&', '{', '[', ']', '|', '_', '\\', '%', '^', '=', '@', '>', '²'


REPLACE_IF_FIRST_CHAR = [' ', '\n']
STR_TO_REPLACE_BY_SINGLE_OCCURENCES = ['!', '.', ' ', '-']
REPLACE_IF_SEQUENCE_IN_IT = [';-)', '~', '""', ':)', ';)', ':-)', '^^']


# preprocessing won't add a dot at the end of recipe's step if the string end by this character.
# ACCEPTED_END_OF_STRING_CHAR = ['.', '!', '?', ':']
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

    ingredients_df.ingredient = ingredients_df.ingredient.apply(lambda x : re.sub(' +', ' ', x))
    ingredients_df.ingredient = ingredients_df.ingredient.apply(lambda x : re.sub('c.à.s', '1 c.à.s', x))
    ingredients_df.ingredient = ingredients_df.ingredient.apply(lambda x : re.sub('c.à.c', '1 c.à.c', x))
    ingredients_df.ingredient = ingredients_df.ingredient.apply(lambda x : re.sub('\' ', '\'', x))
    ingredients_df.ingredient = ingredients_df.ingredient.apply(lambda x : re.sub('\xa0', ' ', x))


    ingredients_df.ingredient = ingredients_df.ingredient.apply(lambda x : x[0].upper()+x[1:])




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
    for i in range(3):
        recipes_df = replace_sequences_in_column(recipes_df, ['—', '–'], '-')
        recipes_df = replace_sequences_in_column(recipes_df, [' .', '…'], '.')
        recipes_df.drop_duplicates(inplace=True)
        recipes_df = recipes_df.reset_index().drop(columns='index')


        # Drop recipes whose lengths are inferior to 20
        recipes_df.recipe_steps = recipes_df.recipe_steps.apply(lambda x : x if len(x) >40 else 'ThisShoulDBeDroPPped')
        recipes_df = drop_rows(recipes_df, recipes_df[recipes_df.recipe_steps=='ThisShoulDBeDroPPped'].index)

        # Deleting problematics rows individually from recipes_df in INDIVIDUALS_RECIPE_ID_TO_DROP
        # recipes_df = recipes_df[recipes_df.recipe_steps.apply(lambda x : \
        #     x.find("Faire fondre le chocolat et le yaourt au micro-ondes, c\'est plus rapide.\n\nQuand c\'est fondu, ajouter l\'oeuf et le sucre, mélanger.")) == -1]
        # recipes_df = recipes_df[recipes_df.recipe_id != 55088]
        for recipe_id in INDIVIDUALS_RECIPE_ID_TO_DROP:
            recipes_df = recipes_df[recipes_df.recipe_id != recipe_id]


        # Enlever les '- ' avec un chiffre derrière éventuellement
        recipes_df.recipe_steps = recipes_df.recipe_steps.replace(to_replace ='(^|\\n) *(\*|-|•|~)+ *\d* *', value = '\g<1>', regex = True)

        # Regex Dangereux testé uniquement sur Regex101
        recipes_df.recipe_steps = recipes_df.recipe_steps.replace(to_replace ='(^|\\n)\d* *(\*|-|\.|•|~|\/|\))+ *\d* *', value = '\g<1>', regex = True)

        # looking for 2) 8) ...
        recipes_df.recipe_steps = recipes_df.recipe_steps.replace(to_replace ='(\\n\\n)\d\)', value = '\g<1>', regex = True)
        # recipes_df = manage_regex_changes(recipes_df)

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
        recipes_df = replace_sequences_in_column(recipes_df, [' .'], '.')


        # Apply capitalize to each steps of the recipe_steps columns
        recipes_df = capitalize_steps(recipes_df)

        # resolving the ',\n\n issue'
        recipes_df.recipe_steps = recipes_df.recipe_steps.apply(lambda x : x.split("\n\n"))
        recipes_df.recipe_steps = recipes_df.recipe_steps.apply(resolving_capital_after_commas_issues)


        # dropping useless columns
        recipes_df.drop(columns='first_letter', inplace=True)

        recipes_df.to_csv('./data/clean_recipes.csv', index=False, header=True)



        ## Individual treatment of characters who didn't got treated properly by functions, e.g. '  ' some double spaces
        recipes_df.recipe_steps = recipes_df.recipe_steps.apply(lambda x : x.replace('  ', ' '))
        # replace '3/ ' by empty string but keep 1/2 litters
        recipes_df.recipe_steps = recipes_df.recipe_steps.replace(to_replace ='(^|\\n)[1-9]*(\/)+\d{0} ', value = '', regex = True)
        recipes_df.recipe_steps = recipes_df.recipe_steps.replace(to_replace ='(^|\\n)[1-9]*(\/)+\d{0} ', value = '', regex = True)

        recipes_df = drop_rows_with_character_in_it(recipes_df, [',\n\n'])


        # recipes_df = recipes_df.reset_index().drop(columns='index')
        recipes_df.recipe_steps = recipes_df.recipe_steps.replace(to_replace ='(\\n\\n)\d\)', value = '\g<1>', regex = True)
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


def replace_multiple_chars_by_a_single_one(df, chars_list=STR_TO_REPLACE_BY_SINGLE_OCCURENCES, iteration=35) :
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
    df.recipe_steps = df.recipe_steps.apply(lambda x : x.split("\n\n"))
    # df.recipe_steps = df.recipe_steps.apply(lambda x : x.strip())
    df.recipe_steps = df.recipe_steps.apply(dot_at_the_end_of_string)

    return df

def capitalize_steps_loop(steps_list):
    return "\n\n".join([step[0].upper()+step[1:] for step in steps_list if step != ''])

def dot_at_the_end_of_string(steps_list):
    modified_steps_list = []
    for step in steps_list:
        stepper = step.strip()
        if stepper != '' :
            if stepper[-1] in  ['.', '!', '?', ':', ';', ',']:
                modified_steps_list.append(stepper)
            else :
                modified_steps_list.append(stepper+'.')
    return "\n\n".join(modified_steps_list)


def resolving_capital_after_commas_issues(steps_list):
    """resolving ,\n\n issues"""
    modified_steps_list = []
    last_step='aa'
    for step in steps_list:

        stepper = step
        if stepper != '' :
            if last_step[-1] in  [',', ';']:
                modified_steps_list.append(stepper[0].lower()+stepper[1:])
            else :
                modified_steps_list.append(stepper)
            if last_step != 'aa' and last_step!='' :
                last_step = stepper

    return "\n\n".join(modified_steps_list)













def manage_regex_changes(df) :
    df.recipe_steps = df.recipe_steps.replace(to_replace ='\n\n\d\)', value = '\n\n', regex = True)
    return df
