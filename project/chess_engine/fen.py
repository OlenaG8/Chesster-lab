PIECES = {
0:"",

1:"P",
2:"N",
3:"B",
4:"R",
5:"Q",
6:"K",

7:"p",
8:"n",
9:"b",
10:"r",
11:"q",
12:"k"
}


def board_to_fen(board):
    rows=[]

    for r in range(8):
        empty = 0
        row = ""
        for c in range(8):
            piece = board[r * 8 + c]

            if piece == 0:
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