# CMPUT 455 Assignment 4 starter code (PoE2)
# Implement the specified commands to complete the assignment
# Full assignment specification on Canvas

import sys
import signal

class CommandInterface:
    # The following is already defined and does not need modification
    # However, you may change or add to this code as you see fit, e.g. adding class variables to init

    def __init__(self):
        # Define the string to function command mapping
        self.command_dict = {
            "help"     : self.help,
            "init_game": self.init_game,   # init_game w h p s [board]
            "show"     : self.show,
            "timelimit": self.timelimit,   # timelimit seconds
            "genmove"  : self.genmove,     # see assignment spec
            "play"     : self.play, 
            "undo"     : self.undo,
            "legal"    : self.legal,
            "score"    : self.score,
            "winner"   : self.winner
        }

        self.board = [[0]]
        self.to_play = 1
        self.handicap = 0.0
        self.score_cutoff = float("inf")
        self.time_limit = 1

    # Convert a raw string to a command and a list of arguments
    def process_command(self, s):
        s = s.lower().strip()
        if len(s) == 0:
            return True
        command = s.split(" ")[0]
        args = [x for x in s.split(" ")[1:] if len(x) > 0]
        if command not in self.command_dict:
            print("? Uknown command.\nType 'help' to list known commands.", file=sys.stderr)
            print("= -1\n")
            return False
        ##Error handling is currently commented out to enable better error messages
        ##You may want to re-enable this for your submission in case there are unknown errors
        #try:
        return self.command_dict[command](args)
        #except Exception as e: 
        #    print("Command '" + s + "' failed with exception:", file=sys.stderr)
        #    print(e, file=sys.stderr)
        #    print("= -1\n")
        #    return False
        
    # Will continuously receive and execute commands
    # Commands should return True on success, and False on failure
    # Every command will print '= 1' or '= -1' at the end of execution to indicate success or failure respectively
    def main_loop(self):
        while True:
            s = input()
            if s.split(" ")[0] == "exit":
                print("= 1\n")
                return True
            if self.process_command(s):
                print("= 1\n")

    # List available commands
    def help(self, args):
        for command in self.command_dict:
            if command != "help":
                print(command)
        print("exit")
        return True

    # Helper function for command argument checking
    # Will make sure there are enough arguments, and that they are valid integers
    def arg_check(self, args, template):
        if len(args) < len(template.split(" ")):
            print("Not enough arguments.\nExpected arguments:", template, file=sys.stderr)
            print("Recieved arguments: ", end="", file=sys.stderr)
            for a in args:
                print(a, end=" ", file=sys.stderr)
            print(file=sys.stderr)
            return False
        for i, arg in enumerate(args):
            try:
                args[i] = int(arg)
            except ValueError:
                try:
                    args[i] = float(arg)
                except ValueError:
                    print("Argument '" + arg + "' cannot be interpreted as a number.\nExpected arguments:", template, file=sys.stderr)
                    return False
        return True
    
    # Command functions needed for playing.
    # Feel free to modify them if needed, but keep their functionality intact.

    def winner(self, args):
        '''
            >> winner
            Prints the winner information.
        '''
        p1_score, p2_score = self.calculate_score()
        if p1_score >= self.score_cutoff:
            print(1)
        elif p2_score >= self.score_cutoff:
            print(2)
        elif len(self.get_moves()) == 0:
            if (p1_score > p2_score):
                print(1)
            else:
                print(2)
        else:
            print("unknown")
        return True

    def valid_move(self, c, r, num):
        return  c >= 0 and c < len(self.board[0]) and\
                r >= 0 and r < len(self.board) and\
                (num == 1 or num == 2) and\
                self.is_pos_avail(c, r)

    def is_pos_avail(self, c, r):
        return self.board[r][c] == 0

    def legal(self, args):
        '''
            >> legal <col> <row>
            Checks if player <player> can play at position (<col>, <row>) on the board.
        '''
        if not self.arg_check(args, "c r"):
            return False
        c, r = [int(arg) for arg in args]
        if self.valid_move(c, r, self.player) and not self.is_game_over():
            print("yes")
        else:
            print("no")
        return True


    def undo(self, args):
        '''
            >> undo
            Undoes the last move committed by the play command.
        '''
        if (len(self.move_history) == 0):
            print("No moves to undo!", file=sys.stderr)
            return False

        # undo the move
        col, row, player = self.move_history.pop()
        self.board[row][col] = None

        # switch player
        self.player = player

        # recompute the score for player
        self.compute_score(int(player))
        return True

    # init_game w h p s
    def init_game(self, args):
        # Check arguments
        if len(args) > 4:
            self.board_str = args.pop()
        else:
            self.board_str = ""
        if not self.arg_check(args, "w h p s"):
            return False
        w, h, p, s = args
        if not (1 <= w <= 20 and 1 <= h <= 20):
            print("Invalid board size:", w, h, file=sys.stderr)
            return False
        
        #Initialize game state
        self.width = w
        self.height = h
        self.handicap = p
        if s == 0:
            self.score_cutoff = float("inf")
        else:
            self.score_cutoff = s
        
        self.board = []
        for r in range(self.height):
            self.board.append([0]*self.width)
        self.to_play = 1
        self.p1_score = 0
        self.p2_score = self.handicap
        return True

    def show(self, args):
        for row in self.board:
            print(" ".join(["_" if v == 0 else str(v) for v in row]))
        return True

    # Sets the timelimit for genmove, non-negative integer
    def timelimit(self, args):
        if not self.arg_check(args, "t"):
            return False

        self.time_limit = int(args[0])
        return True

    def play(self, args):
        if not self.arg_check(args, "x y"):
            return False
        
        try:
            x = int(args[0])
            y = int(args[1])
        except ValueError:
            print("Illegal move: " + " ".join(args), file=sys.stderr)
            return False
        
        if not (0 <= x < self.width) or not (0 <= y < self.height) or self.board[y][x] != 0:
            print("Illegal move: " + " ".join(args), file=sys.stderr)
            return False
        
        if self.p1_score >= self.score_cutoff or self.p2_score >= self.score_cutoff:
            print("Illegal move: " + " ".join(args), "game ended.", file=sys.stderr)
            return False
        
        # put the piece onto the board
        self.make_move(x, y)

        return True

    def score(self, args):
        p1_score, p2_score = self.calculate_score()
        print(p1_score, p2_score)
        return True
    
    # Optional helper functions that you may use or replace with your own.

    def get_moves(self):
        moves = []
        for y in range(self.height):
            for x in range(self.width):
                if self.board[y][x] == 0:
                    moves.append((x, y))
        moves.sort(key=lambda x: abs(3-x[0])+abs(3-x[1]))
        return moves

    def make_move(self, x, y):
        self.board[y][x] = self.to_play
        if self.to_play == 1:
            self.to_play = 2
        else:
            self.to_play = 1

    def undo_move(self, x, y):
        self.board[y][x] = 0
        if self.to_play == 1:
            self.to_play = 2
        else:
            self.to_play = 1

    # Returns p1_score, p2_score
    def calculate_score(self):
        p1_score = 0
        p2_score = self.handicap

        # Progress from left-to-right, top-to-bottom
        # We define lines to start at the topmost (and for horizontal lines leftmost) point of that line
        # At each point, score the lines which start at that point
        # By only scoring the starting points of lines, we never score line subsets
        for y in range(self.height):
            for x in range(self.width):
                c = self.board[y][x]
                if c != 0:
                    lone_piece = True # Keep track of the special case of a lone piece
                    # Horizontal
                    hl = 1
                    if x == 0 or self.board[y][x-1] != c: #Check if this is the start of a horizontal line
                        x1 = x+1
                        while x1 < self.width and self.board[y][x1] == c: #Count to the end
                            hl += 1
                            x1 += 1
                    else:
                        lone_piece = False
                    # Vertical
                    vl = 1
                    if y == 0 or self.board[y-1][x] != c: #Check if this is the start of a vertical line
                        y1 = y+1
                        while y1 < self.height and self.board[y1][x] == c: #Count to the end
                            vl += 1
                            y1 += 1
                    else:
                        lone_piece = False
                    # Diagonal
                    dl = 1
                    if y == 0 or x == 0 or self.board[y-1][x-1] != c: #Check if this is the start of a diagonal line
                        x1 = x+1
                        y1 = y+1
                        while x1 < self.width and y1 < self.height and self.board[y1][x1] == c: #Count to the end
                            dl += 1
                            x1 += 1
                            y1 += 1
                    else:
                        lone_piece = False
                    # Anit-diagonal
                    al = 1
                    if y == 0 or x == self.width-1 or self.board[y-1][x+1] != c: #Check if this is the start of an anti-diagonal line
                        x1 = x-1
                        y1 = y+1
                        while x1 >= 0 and y1 < self.height and self.board[y1][x1] == c: #Count to the end
                            al += 1
                            x1 -= 1
                            y1 += 1
                    else:
                        lone_piece = False
                    # Add scores for found lines
                    for line_length in [hl, vl, dl, al]:
                        if line_length > 1:
                            if c == 1:
                                p1_score += 2 ** (line_length-1)
                            else:
                                p2_score += 2 ** (line_length-1)
                    # If all found lines are length 1, check if it is the special case of a lone piece
                    if hl == vl == dl == al == 1 and lone_piece:
                        if c == 1:
                            p1_score += 1
                        else:
                            p2_score += 1
        return p1_score, p2_score
    
    # Returns terminal, relative score
    def get_relative_score(self):
        p1_score, p2_score = self.calculate_score()
        if self.to_play == 1:
            score = p1_score - p2_score
        else:
            score = p2_score - p1_score
        if p1_score >= self.score_cutoff or p2_score >= self.score_cutoff:
            return True, score
        else:
            # Check if the board is full
            for y in range(self.height):
                for x in range(self.width):
                    if self.board[y][x] == 0:
                        return False, score
            return True, score

    def negamax_alpha_beta_limited_depth(self, alpha, beta, depth, max_depth):
        #print(depth, max_depth)
        #Check if the position is in the transposition table
        best_move = None
        hash = str(self.board)
        #If the position is already solved, return the solved value
        if hash in self.tt:
            value, valid, best_move = self.tt[hash]
            if valid:
                return value, True
        #Check if the position is terminal
        terminal, score = self.get_relative_score()
        if terminal:
            self.tt[hash] = (score, True, None)
            return score, True
        #Check if we've reached the max depth
        if depth == max_depth:
            self.tt[hash] = (score, False, None)
            return score, False
        #If there is a recorded best move in the tt, search that first
        moves = self.get_moves()
        if best_move is not None:
            moves.remove(best_move)
            moves.insert(0,best_move)
        #Check all legal moves until a cutoff
        valid_result = True
        value = -float('inf')
        best_found_move = None
        for move in moves:
            #Recursive call
            self.make_move(*move)
            child_value, valid_child = self.negamax_alpha_beta_limited_depth(-beta, -alpha, depth+1, max_depth)
            self.undo_move(*move)
            #Update value and record the best move found
            if -child_value > value:
                value = -child_value
                best_found_move = move
            #Record whether we encounter any heuristic values
            valid_result = valid_result and valid_child
            #Check for alpha beta cutoffs
            alpha = max(alpha, value)
            if alpha >= beta:
                self.tt[hash] = (value, valid_result, best_found_move)
                return value, valid_result
        #No cutoffs, return result
        self.tt[hash] = (value, valid_result, best_found_move)
        return value, valid_result

    # To implement for assignment 4.
    # Make sure you print a move within the specified time limit (1 second by default)
    # Print the x and y coordinates of your chosen move, space separated.
    def genmove(self, args):
        class TimeoutException(Exception):
            pass
        def handler(signum, frame):
            raise TimeoutException("Function timed out.")
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(self.time_limit)
        self.tt = {}
        max_depth = 0
        valid = False
        org_board = [list(row) for row in self.board]
        org_to_play = self.to_play
        try:
            while not valid:
                value, valid = self.negamax_alpha_beta_limited_depth(-float('inf'), float('inf'), 0, max_depth)
                max_depth += 1
        except TimeoutException:
            pass
        signal.alarm(0)
        self.board = org_board
        self.to_play = org_to_play
        best_move = self.tt[str(self.board)][2]
        self.make_move(*best_move)
        print(best_move[0], best_move[1])
        return True

if __name__ == "__main__":
    interface = CommandInterface()
    interface.main_loop()
