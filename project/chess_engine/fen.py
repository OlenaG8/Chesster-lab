PIECES = {
    0: "b",       # black_bishop
    1: "k",       # black_king
    2: "n",       # black_knight
    3: "p",       # black_pawn
    4: "q",       # black_queen
    5: "r",       # black_rook
    6: "EMPTY",   # empty
    7: "B",       # white_bishop
    8: "K",       # white_king
    9: "N",       # white_knight
    10: "P",      # white_pawn
    11: "Q",      # white_queen
    12: "R",      # white_rook
}


def board_to_fen(board):
    rows = []

    for r in range(8):
        empty = 0
        row = ""

        for c in range(8):
            piece = board[r * 8 + c]

            if piece == 6:
                empty += 1
            else:
                if empty:
                    row += str(empty)
                    empty = 0
                row += PIECES[piece]

        if empty:
            row += str(empty)

        rows.append(row)
    return "/".join(rows)