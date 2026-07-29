import chess.engine


class Stockfish:
    def __init__(self, path):
        self.engine = (chess.engine.SimpleEngine.popen_uci(path))

    def get_move(self, board):
        result = self.engine.play(board, chess.engine.Limit(time=1))
        return result.move

    def close(self):
        self.engine.quit()