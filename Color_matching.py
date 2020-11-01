import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline


def make_color_wheel():
    colourWheel = []

    colourWheel.append(["Red", 255, 40, 0])
    # changed red to include a better diff between orange and red
    colourWheel.append(["Orange", 255, 127, 0])
    colourWheel.append(["Yellow", 255, 255, 0])
    colourWheel.append(["Green-Yellow", 127, 255, 0])
    colourWheel.append(["Green", 0, 255, 0])
    colourWheel.append(["Green-Cyan", 0, 255, 127])
    colourWheel.append(["Cyan", 0, 255, 25])
    colourWheel.append(["Azure", 0, 127, 255])
    colourWheel.append(["Blue", 0, 0, 255])
    colourWheel.append(["Violet", 127, 0, 255])
    colourWheel.append(["Magenta", 255, 0, 255])
    colourWheel.append(["Rose", 255, 0, 127])

    return colourWheel

color_wheel=make_color_wheel()
color_wheel

# Calculate difference
def get_name(color_wheel, r_ref, g_ref, b_ref):
    # check neutral colors: all RGB values equal, here I allow a max mean absolute deviation of 10 across the 3 channels
    mean = (r_ref + g_ref + b_ref) / 3
    abs_dev = abs(r_ref - mean) + abs(g_ref - mean) + abs(b_ref - mean)
    mad = abs_dev / 3

    if mad < 10:
        if r_ref < 80:
            color_name = "Black"
        elif r_ref > 200:
            color_name = "White"
        else:
            color_name = "Gray"

    else:

        skin_check = (r_ref in range(165, 241)) and (g_ref in range(125, 221)) and (b_ref in range(110, 201))

        if skin_check == True:
            color_name = "Skin"
        else:

            distances = []
            for c in color_wheel:
                r1 = c[1]
                g1 = c[2]
                b1 = c[3]

                diff = np.sqrt((r_ref - r1) ** 2 + (g_ref - g1) ** 2 + (b_ref - b1) ** 2)
                distances.append(diff)

            # output name of color with smallest difference
            minpos = distances.index(min(distances))

            color_name = color_wheel[minpos][0]

    if color_name == "Orange":
        # check if it is really orange
        orange_check = (r_ref in range(235, 256)) and (g_ref in range(110, 171)) and (b_ref in range(0, 51))
        if orange_check == True:
            color_name = "Orange"
        else:
            color_name = "Brown"

    return color_name

# Apply on extracted colors from images
all_data=pd.read_csv("all_year_colors.csv")

colors=[1,2,3]
names=list()

for n_colors in colors:
    # Create name
    names = 'color' + str(n_colors)

    globals()[names] = []

    for i in range(0, len(all_data)):
        r = all_data.iloc[i]["R1"]
        g = all_data.iloc[i]["G1"]
        b = all_data.iloc[i]["B1"]

    color_name = get_name(color_wheel, r, g, b)

    globals()[names].append(color_name)

colors=['color1','color2','color3']

for col in enumerate(colors):
    all_data[col]=eval(col)

all_data.to_csv("transformed_colors.csv", index=False)





















