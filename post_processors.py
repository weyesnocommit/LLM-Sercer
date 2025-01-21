import re

### t5 mihm
def t5_mihm(text:str):
    if text:
        return text[2:-2]
    return None

### recipmetning
def parse_generated_recipes(recipe_list):
    parsed_recipes = []
    for recipe in recipe_list:
        recipe_obj = {}
        title_match = re.search(r" title: (.+?)\n", recipe)
        ingredients_match = re.search(r" ingredients: (.+?)\n directions:", recipe)
        directions_match = re.search(r" directions: (.+)", recipe)

        if title_match:
            recipe_obj["title"] = title_match.group(1).strip()
        if ingredients_match:
            ingredients = ingredients_match.group(1).strip().split('--')
            recipe_obj["ingredients"] = [ingredient.strip() for ingredient in ingredients]
        if directions_match:
            directions = directions_match.group(1).strip().split('--')
            recipe_obj["directions"] = [direction.strip() for direction in directions]

        parsed_recipes.append(recipe_obj)
    return parsed_recipes


def to_string(parsed_recipes):
    try:
        title = ""
        ingridients = ""
        direct = ""
        for recipe in parsed_recipes:
            title = recipe['title']
            for ing in recipe['ingredients']:
                ingridients += ing+", "
            for dir in recipe['directions']:
                direct += dir+"\n"
        return title +"\n"+ ingridients + "\n" + direct
    except:
        return parsed_recipes

def gen_as_str(text:str):
    return to_string(parse_generated_recipes(text))



POST_PORCESSORS = {
    "t5-mihm": t5_mihm,
    "t5-recipe-generation": gen_as_str
}