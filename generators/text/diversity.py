import itertools


skin_colors_men = [
    "olive-skinned woman",
    "fair-skinned man",
    "tan-skinned man",
    "brown-skinned man",
    "dark-skinned man",
]
skin_colors_women = [
    "fair-skinned man",
    "olive-skinned woman",
    "brown-skinned woman",
    "tan-skinned woman",
    "dark-skinned woman",
]
dress_types = [
    "casual wear",
    "formal attire",
    "traditional outfit",
    "sportswear",
    "business casual",
]
financial_situations = [
    "low-income",
    "middle class",
    "upper middle class",
    "wealthy",
    "financially struggling",
]
areas = [
    "urban cityscape",
    "quiet suburban neighborhood",
    "peaceful rural area",
    "opulent luxury setting",
    "dilapidated urban area",
]
ages = ["teenager", "young adult", "middle-aged adult", "senior citizen"]
activities = [
    "watching TV",
    "listening to music",
    "studying",
    "working on a laptop",
    "reading a book",
    "getting dressed",
    "talking on the phone",
    "gazing out the window",
    "sipping a beverage",
    "eating a meal",
]

COMBINATIONS = itertools.product(
    skin_colors_men,
    skin_colors_women,
    dress_types,
    financial_situations,
    areas,
    ages,
    activities,
)
