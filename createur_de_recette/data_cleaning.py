

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
