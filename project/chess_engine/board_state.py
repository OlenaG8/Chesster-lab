import chess


class BoardState:
    def __init__(self):
        self.board = chess.Board()

    def update_fen(self,fen):
        self.board = chess.Board(fen)

    def legal_move(self,uci):
        move = chess.Move.from_uci(uci)
        return move in self.board.legal_moves

    def push(self,uci):
        move = chess.Move.from_uci(uci)
        self.board.push(move)