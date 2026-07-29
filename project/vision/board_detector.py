FILES = "abcdefgh"
RANKS = "87654321"


def split_board(board_img):
    size = board_img.shape[0] // 8

    squares = {}

    for row in range(8):
        for col in range(8):

            name = (FILES[col] + RANKS[row])

            crop = board_img[
                row*size:(row+1)*size,
                col*size:(col+1)*size
            ]

            squares[name] = crop

    return squares