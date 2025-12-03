
# CMPUT 455 Assignment 4 - Optimized PoE2 Player (Alpha-Beta with Heuristics)
import sys
import time
import random

class CommandInterface:
    def __init__(self):
        self.command_dict = {
            "help"     : self.help,
            "init_game": self.init_game,
            "show"     : self.show,
            "timelimit": self.timelimit,
            "genmove"  : self.genmove,
            "play"     : self.play,
            "score"    : self.score,
        }

        self.board = [[0]]
        self.to_play = 1
        self.handicap = 0.0
        self.score_cutoff = float("inf")
        self.time_limit = 1
        self.tt = {}
        self.move_ordering_history = {}
        self.killer_moves = {}
        self.zobrist_table = None

    # ======== Main Loop & Utilities ========

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
        return self.command_dict[command](args)

    def main_loop(self):
        while True:
            s = input()
            if s.split(" ")[0] == "exit":
                print("= 1\n")
                return True
            if self.process_command(s):
                print("= 1\n")

    def help(self, args):
        for command in self.command_dict:
            if command != "help":
                print(command)
        print("exit")
        return True

    def arg_check(self, args, template):
        needed = len(template.split(" "))
        if len(args) < needed:
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

    # ======== Zobrist Hashing ========

    def init_zobrist(self):
        """Initialize Zobrist hashing table."""
        random.seed(42)
        self.zobrist_table = {}
        for y in range(self.height):
            for x in range(self.width):
                for player in [1, 2]:
                    self.zobrist_table[(x, y, player)] = random.getrandbits(64)
        self.zobrist_player = [random.getrandbits(64), random.getrandbits(64)]

    def compute_zobrist_hash(self):
        """Compute Zobrist hash for current board state."""
        h = 0
        for y in range(self.height):
            row = self.board[y]
            for x in range(self.width):
                v = row[x]
                if v != 0:
                    h ^= self.zobrist_table[(x, y, v)]
        h ^= self.zobrist_player[self.to_play - 1]
        return h

    # ======== Game Commands ========

    def init_game(self, args):
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

        self.width = w
        self.height = h
        self.handicap = p
        if s == 0:
            self.score_cutoff = float("inf")
        else:
            self.score_cutoff = s

        self.board = [[0 for _ in range(self.width)] for _ in range(self.height)]
        self.to_play = 1
        self.p1_score = 0
        self.p2_score = self.handicap
        self.tt = {}
        self.move_ordering_history = {}
        self.killer_moves = {}

        # Apply board string if provided
        if self.board_str:
            nonzero = 0
            if len(self.board_str) != self.width * self.height:
                print("Invalid board string length", file=sys.stderr)
                return False
            for y in range(self.height):
                for x in range(self.width):
                    c = self.board_str[y * self.width + x]
                    if c == '1':
                        self.board[y][x] = 1
                        nonzero += 1
                    elif c == '2':
                        self.board[y][x] = 2
                        nonzero += 1
                    elif c != '.':
                        print("Invalid board string character:", c, file=sys.stderr)
                        return False
            self.to_play = 1 if nonzero % 2 == 0 else 2

        # Initialize Zobrist hashing
        self.init_zobrist()
        self.current_hash = self.compute_zobrist_hash()

        return True

    def show(self, args):
        for row in self.board:
            print(" ".join(["_" if v == 0 else str(v) for v in row]))
        return True

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
        except (ValueError, TypeError):
            print("Illegal move: " + " ".join([str(a) for a in args]), file=sys.stderr)
            return False

        if not (0 <= x < self.width) or not (0 <= y < self.height) or self.board[y][x] != 0:
            print("Illegal move: " + " ".join([str(a) for a in args]), file=sys.stderr)
            return False

        p1_score, p2_score = self.calculate_score()
        if p1_score >= self.score_cutoff or p2_score >= self.score_cutoff:
            print("Illegal move: " + " ".join([str(a) for a in args]), "game ended.", file=sys.stderr)
            return False

        self.make_move(x, y)
        return True

    def score(self, args):
        p1_score, p2_score = self.calculate_score()
        print(p1_score, p2_score)
        return True

    # ======== Board / Move Operations ========

    def get_moves(self):
        """
        Move generator with strong pruning:
        - If board has stones: mostly return moves adjacent to any stone,
          plus a few extra near center.
        - If empty: return a few central openings.
        """
        H, W = self.height, self.width
        board = self.board

        all_empty = []
        has_stone = False

        for y in range(H):
            row = board[y]
            for x in range(W):
                if row[x] == 0:
                    all_empty.append((x, y))
                else:
                    has_stone = True

        # Empty board: restrict to central region
        if not has_stone:
            cx, cy = W // 2, H // 2
            all_empty.sort(key=lambda m: abs(m[0] - cx) + abs(m[1] - cy))
            # Keep only a few best openings
            return all_empty[:min(10, len(all_empty))]

        # Non-empty board: prefer adjacency to any stone
        adjacent_moves = []
        adj_set = set()
        for (x, y) in all_empty:
            found = False
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < W and 0 <= ny < H and board[ny][nx] != 0:
                        found = True
                        break
                if found:
                    break
            if found:
                adjacent_moves.append((x, y))
                adj_set.add((x, y))

        # If adjacency list is big enough, just use it
        if len(adjacent_moves) >= 10 or len(all_empty) <= 10:
            return adjacent_moves if adjacent_moves else all_empty

        # Otherwise, add a few central non-adjacent moves
        cx, cy = W // 2, H // 2
        remaining = [m for m in all_empty if m not in adj_set]
        remaining.sort(key=lambda m: abs(m[0] - cx) + abs(m[1] - cy))
        extra_needed = max(0, 10 - len(adjacent_moves))
        extra_moves = remaining[:extra_needed]
        return adjacent_moves + extra_moves

    def make_move(self, x, y):
        """Make a move and update Zobrist hash."""
        player = self.to_play
        self.board[y][x] = player
        self.current_hash ^= self.zobrist_table[(x, y, player)]
        self.current_hash ^= self.zobrist_player[player - 1]
        self.to_play = 2 if player == 1 else 1
        self.current_hash ^= self.zobrist_player[self.to_play - 1]

    def undo_move(self, x, y):
        """Undo a move and restore Zobrist hash."""
        # Undo player toggle in hash
        self.current_hash ^= self.zobrist_player[self.to_play - 1]
        self.to_play = 2 if self.to_play == 1 else 1
        self.current_hash ^= self.zobrist_player[self.to_play - 1]
        # Remove stone
        self.current_hash ^= self.zobrist_table[(x, y, self.to_play)]
        self.board[y][x] = 0

    # ======== Scoring & Evaluation ========

    def calculate_score(self):
        """
        Correct PoE2 scoring:
        - Lines of length L >= 2 are worth 2^L, counted once per maximal line.
        - Lone stones are worth 1.
        - Player 2 starts with handicap.
        """
        p1_score = 0
        p2_score = self.handicap
        H, W = self.height, self.width
        board = self.board

        for y in range(H):
            for x in range(W):
                c = board[y][x]
                if c == 0:
                    continue

                lone_piece = True

                # Horizontal
                hl = 1
                if x == 0 or board[y][x - 1] != c:
                    x1 = x + 1
                    while x1 < W and board[y][x1] == c:
                        hl += 1
                        x1 += 1
                else:
                    lone_piece = False

                # Vertical
                vl = 1
                if y == 0 or board[y - 1][x] != c:
                    y1 = y + 1
                    while y1 < H and board[y1][x] == c:
                        vl += 1
                        y1 += 1
                else:
                    lone_piece = False

                # Diagonal
                dl = 1
                if y == 0 or x == 0 or board[y - 1][x - 1] != c:
                    x1, y1 = x + 1, y + 1
                    while x1 < W and y1 < H and board[y1][x1] == c:
                        dl += 1
                        x1 += 1
                        y1 += 1
                else:
                    lone_piece = False

                # Anti-diagonal
                al = 1
                if y == 0 or x == W - 1 or board[y - 1][x + 1] != c:
                    x1, y1 = x - 1, y + 1
                    while x1 >= 0 and y1 < H and board[y1][x1] == c:
                        al += 1
                        x1 -= 1
                        y1 += 1
                else:
                    lone_piece = False

                longest = max(hl, vl, dl, al)
                if longest > 1:
                    if c == 1:
                        p1_score += 2 ** longest
                    else:
                        p2_score += 2 ** longest
                elif lone_piece:
                    if c == 1:
                        p1_score += 1
                    else:
                        p2_score += 1

        return p1_score, p2_score

    def evaluate_threats(self):
        """
        Heuristic evaluation:
        - Base: score difference from current player perspective.
        - Threats: line-extension potential for both players (weighted).
        """
        p1_score, p2_score = self.calculate_score()
        base_score = p1_score - p2_score if self.to_play == 1 else p2_score - p1_score

        H, W = self.height, self.width
        board = self.board
        threat_score = 0

        # For each empty cell, consider how strong a line extension it would be.
        for y in range(H):
            for x in range(W):
                if board[y][x] != 0:
                    continue
                for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
                    for player in (1, 2):
                        line_len = 1
                        # Positive direction
                        nx, ny = x + dx, y + dy
                        while 0 <= nx < W and 0 <= ny < H and board[ny][nx] == player:
                            line_len += 1
                            nx += dx
                            ny += dy
                        # Negative direction
                        nx, ny = x - dx, y - dy
                        while 0 <= nx < W and 0 <= ny < H and board[ny][nx] == player:
                            line_len += 1
                            nx -= dx
                            ny -= dy

                        if line_len >= 2:
                            potential = line_len * line_len
                            if (player == 1 and self.to_play == 1) or (player == 2 and self.to_play == 2):
                                threat_score += potential
                            else:
                                threat_score -= potential

        return base_score + 0.15 * threat_score

    def get_relative_score(self):
        """Return (terminal, score-from-current-player-perspective)."""
        p1_score, p2_score = self.calculate_score()
        if self.to_play == 1:
            score = p1_score - p2_score
        else:
            score = p2_score - p1_score

        if p1_score >= self.score_cutoff or p2_score >= self.score_cutoff:
            return True, score

        # Check full board
        for y in range(self.height):
            for x in range(self.width):
                if self.board[y][x] == 0:
                    return False, score
        return True, score

    # ======== Move Ordering & Search ========

    def order_moves(self, moves, depth, best_move):
        """Order moves: TT move, killer moves, then heuristic scoring."""
        if not moves:
            return []

        ordered = []
        moves_set = set(moves)

        # 1. Best move from TT
        if best_move and best_move in moves_set:
            ordered.append(best_move)
            moves_set.remove(best_move)

        # 2. Killer moves (up to 2)
        if depth in self.killer_moves:
            for killer in self.killer_moves[depth][:2]:
                if killer in moves_set:
                    ordered.append(killer)
                    moves_set.remove(killer)

        # 3. Score remaining moves
        center_x, center_y = self.width // 2, self.height // 2
        scored_moves = []
        board = self.board
        player = self.to_play
        opp = 2 if player == 1 else 1

        for move in moves_set:
            x, y = move
            score = self.move_ordering_history.get(move, 0) * 2

            # Center bonus
            dist = abs(x - center_x) + abs(y - center_y)
            score += (7 - dist) * 100

            # Adjacency
            adjacent_score = 0
            for dx, dy in (
                (-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (1, 1), (-1, 1), (1, -1)
            ):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    v = board[ny][nx]
                    if v == player:
                        adjacent_score += 3
                    elif v == opp:
                        adjacent_score += 1
            score += adjacent_score * 150

            # Line extension potential
            for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
                # own lines
                line_len = 0
                for direction in (1, -1):
                    nx, ny = x + dx * direction, y + dy * direction
                    while 0 <= nx < self.width and 0 <= ny < self.height and board[ny][nx] == player:
                        line_len += 1
                        nx += dx * direction
                        ny += dy * direction
                if line_len > 0:
                    score += (line_len ** 2) * 500

                # opponent lines to block
                opp_len = 0
                for direction in (1, -1):
                    nx, ny = x + dx * direction, y + dy * direction
                    while 0 <= nx < self.width and 0 <= ny < self.height and board[ny][nx] == opp:
                        opp_len += 1
                        nx += dx * direction
                        ny += dy * direction
                if opp_len > 0:
                    score += (opp_len ** 2) * 300

            scored_moves.append((score, move))

        scored_moves.sort(reverse=True)
        ordered.extend([m for _, m in scored_moves])
        return ordered

    def negamax_alpha_beta(self, alpha, beta, depth, max_depth):
        """Negamax with alpha-beta pruning and TT."""
        hash_key = self.current_hash

        # Transposition table lookup
        if hash_key in self.tt:
            tt_value, tt_flag, tt_best_move, tt_depth = self.tt[hash_key]
            if tt_depth >= max_depth - depth:
                if tt_flag == 'exact':
                    return tt_value, True, tt_best_move
                elif tt_flag == 'lower' and tt_value >= beta:
                    return tt_value, True, tt_best_move
                elif tt_flag == 'upper' and tt_value <= alpha:
                    return tt_value, True, tt_best_move

        # Terminal check
        terminal, score = self.get_relative_score()
        if terminal:
            # Add depth-sensitive bonus so faster wins / slower losses are preferred
            if score > 0:
                score += 1000 - depth
            elif score < 0:
                score -= 1000 - depth
            self.tt[hash_key] = (score, 'exact', None, max_depth - depth)
            return score, True, None

        # Depth limit
        if depth >= max_depth:
            eval_score = self.evaluate_threats()
            self.tt[hash_key] = (eval_score, 'exact', None, 0)
            return eval_score, False, None

        moves = self.get_moves()
        if not moves:
            return 0, True, None

        # Order moves
        tt_best = self.tt.get(hash_key, (None, None, None, None))[2]
        moves = self.order_moves(moves, depth, tt_best)

        # Beam-style restriction: keep only top K moves depending on depth
        if depth == 0:
            max_moves = 10
        elif depth == 1:
            max_moves = 8
        else:
            max_moves = 6
        moves = moves[:max_moves]

        value = -float('inf')
        best_found_move = None
        valid_result = True
        original_alpha = alpha

        for move in moves:
            self.make_move(*move)
            child_value, valid_child, _ = self.negamax_alpha_beta(-beta, -alpha, depth + 1, max_depth)
            self.undo_move(*move)

            child_value = -child_value

            if child_value > value:
                value = child_value
                best_found_move = move

            valid_result = valid_result and valid_child
            alpha = max(alpha, value)

            if alpha >= beta:
                # Alpha-beta cutoff: update killer + history
                if depth not in self.killer_moves:
                    self.killer_moves[depth] = []
                if move not in self.killer_moves[depth]:
                    self.killer_moves[depth].insert(0, move)
                    if len(self.killer_moves[depth]) > 2:
                        self.killer_moves[depth].pop()

                self.move_ordering_history[move] = self.move_ordering_history.get(move, 0) + (max_depth - depth) ** 2
                self.tt[hash_key] = (value, 'lower', best_found_move, max_depth - depth)
                return value, valid_result, best_found_move

        # Store in TT
        if value <= original_alpha:
            flag = 'upper'
        elif value >= beta:
            flag = 'lower'
        else:
            flag = 'exact'
        self.tt[hash_key] = (value, flag, best_found_move, max_depth - depth)
        return value, valid_result, best_found_move

    # ======== Move Generation (Iterative Deepening) ========

    def genmove(self, args):
        """Generate best move with iterative deepening alpha-beta."""
        start_time = time.time()
        # Use a bit less than timelimit for safety
        time_budget = self.time_limit * 0.85

        max_depth = 1
        best_move = None
        best_value = -float('inf')

        self.killer_moves = {}

        moves = self.get_moves()
        if not moves:
            print("0 0")
            return True

        # Fallback: central-ish move
        center_x, center_y = self.width // 2, self.height // 2
        best_move = min(moves, key=lambda m: abs(m[0] - center_x) + abs(m[1] - center_y))

        try:
            while max_depth <= 15:
                elapsed = time.time() - start_time

                # If we are getting late, bail before starting a new depth
                if max_depth > 4 and elapsed > time_budget * 0.6:
                    break

                # Rough next-depth estimate
                if max_depth > 3:
                    estimated_next = elapsed * 3.5
                    if elapsed + estimated_next > time_budget:
                        break

                # Aspiration windows for deeper searches
                if max_depth > 3 and best_value != -float('inf'):
                    window = 100
                    alpha = best_value - window
                    beta = best_value + window
                    value, valid, move = self.negamax_alpha_beta(alpha, beta, 0, max_depth)
                    if value <= alpha or value >= beta:
                        # Re-search full window
                        value, valid, move = self.negamax_alpha_beta(-float('inf'), float('inf'), 0, max_depth)
                else:
                    value, valid, move = self.negamax_alpha_beta(-float('inf'), float('inf'), 0, max_depth)

                elapsed = time.time() - start_time
                if move is not None:
                    best_move = move
                    best_value = value

                # If we're very close to budget, stop here
                if elapsed > time_budget * 0.9:
                    break

                # If result is exact/fully valid at this depth, we can also stop
                if valid:
                    break

                max_depth += 1

        except Exception as e:
            print(f"Error in search: {e}", file=sys.stderr)

        if best_move:
            self.make_move(*best_move)
            print(best_move[0], best_move[1])
        else:
            print("0 0")

        return True


if __name__ == "__main__":
    interface = CommandInterface()
    interface.main_loop()

