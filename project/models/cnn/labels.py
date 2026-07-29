CLASSES = [
    "empty",

    "white_pawn",
    "white_knight",
    "white_bishop",
    "white_rook",
    "white_queen",
    "white_king",

    "black_pawn",
    "black_knight",
    "black_bishop",
    "black_rook",
    "black_queen",
    "black_king",
]

CLASS_TO_ID = {
    name: i
    for i, name in enumerate(CLASSES)
}

ID_TO_CLASS = {
    i: name
    for i, name in enumerate(CLASSES)
}