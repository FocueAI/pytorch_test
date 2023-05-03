def colorful(obj, color="green", display_type="plain"):
    color_dict = {"black": "30", "red": "31", "green": "32", "yellow": "33",
                  "blue": "34", "purple": "35", "cyan": "36", "white": "37"}
    display_type_dict = {"plain": "0", "highlight": "1", "underline": "4",
                         "shine": "5", "inverse": "7", "invisible": "8"}
    s = str(obj)
    color_code = color_dict.get(color, "")
    display = display_type_dict.get(display_type, "")
    out = '\033[{};{}m'.format(display, color_code) + s + '\033[0m'
    return out

out= colorful('hello-colorful-world',color='shine',display_type='underline')
print(out)