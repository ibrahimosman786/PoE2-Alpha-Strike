#-----------------------------------------------------------
#-----------------------------------------------------------
# Name : Ibrahim 
# Student ID : 1868521
# CMPUT 455 Assignment 4 - Ultimate Optimized PoE2 Player
#-----------------------------------------------------------
#-----------------------------------------------------------
import sys
import time
import random

# Precompute powers of 2: POW2[i] = 2^i
POW2 = [1 << i for i in range(25)]

# TT flags
EXACT, LOWER, UPPER = 0, 1, 2


class CommandInterface:
    def __init__(self):
        self.command_dict = {
            "help": self.help,
            "init_game": self.init_game,
            "show": self.show,
            "timelimit": self.timelimit,
            "genmove": self.genmove,
            "play": self.play,
            "score": self.score
        }
        self.board = None
        self.to_play = 1
        self.handicap = 0.0
        self.score_cutoff = float("inf")
        self.time_limit = 1
        self.width = 7
        self.height = 7

        # Using "Global" TT reused across moves in a game
        self.tt = {}
        self.score_cache = {}

    # ---------- Command plumbing ----------

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
        try:
            return self.command_dict[command](args)
        except Exception as e:
            print("Command '" + s + "' failed with exception:", file=sys.stderr)
            print(e, file=sys.stderr)
            print("= -1\n")
            return False

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
        if len(args) < len(template.split(" ")):
            print("Not enough arguments.\nExpected arguments:", template, file=sys.stderr)
            return False
        for i, arg in enumerate(args):
            try:
                args[i] = int(arg)
            except ValueError:
                try:
                    args[i] = float(arg)
                except ValueError:
                    print("Argument '" + arg + "' cannot be interpreted as a number.", file=sys.stderr)
                    return False
        return True

    # ---------- Game setup ----------

    def init_game(self, args):
        if len(args) > 4:
            args.pop()
        if not self.arg_check(args, "w h p s"):
            return False
        w, h, p, s = args
        if not (1 <= w <= 20 and 1 <= h <= 20):
            print("Invalid board size:", w, h, file=sys.stderr)
            return False

        self.width = w
        self.height = h
        self.handicap = p
        self.score_cutoff = float("inf") if s == 0 else s

        self.board = [0] * (w * h)
        self.to_play = 1

        # Empty cells
        self.empty_set = set(range(w * h))

        # Zobrist hashing
        random.seed(455)
        self.zobrist = [[random.getrandbits(64) for _ in range(3)] for _ in range(w * h)]
        self.zobrist_turn = random.getrandbits(64)
        self.current_hash = 0

        # Precompute center scores (for ordering)
        cx, cy = w // 2, h // 2
        self.center_score = []
        for i in range(w * h):
            x, y = i % w, i // w
            dist = abs(x - cx) + abs(y - cy)
            # Strong preference for central moves
            self.center_score.append(200 - dist * 20)

        # Directions: horizontal, vertical , diagonal , anti-diagonal ....
        self.dirs = [(1, 0), (0, 1), (1, 1), (1, -1)]

        # Precomputing the line cells for each position and direction
        self._precompute_lines()

        # New game: clearing TT and score cache
        self.tt = {}
        self.score_cache = {}

        return True

    def _precompute_lines(self):
        """Precompute cells in each direction for every index."""
        w, h = self.width, self.height
        self.line_cells = []
        for idx in range(w * h):
            x, y = idx % w, idx // w
            idx_lines = []
            for dx, dy in self.dirs:
                neg_cells = []
                pos_cells = []
                nx, ny = x - dx, y - dy
                while 0 <= nx < w and 0 <= ny < h:
                    neg_cells.append(ny * w + nx)
                    nx -= dx
                    ny -= dy
                nx, ny = x + dx, y + dy
                while 0 <= nx < w and 0 <= ny < h:
                    pos_cells.append(ny * w + nx)
                    nx += dx
                    ny += dy
                idx_lines.append((tuple(neg_cells), tuple(pos_cells)))
            self.line_cells.append(tuple(idx_lines))

    # ---------- I/O commands ----------

    def show(self, args):
        w = self.width
        for y in range(self.height):
            row = []
            base = y * w
            for x in range(w):
                v = self.board[base + x]
                row.append("_" if v == 0 else str(v))
            print(" ".join(row))
        return True

    def timelimit(self, args):
        if not self.arg_check(args, "t"):
            return False
        self.time_limit = int(args[0])
        return True

    def play(self, args):
        if not self.arg_check(args, "x y"):
            return False
        x, y = int(args[0]), int(args[1])
        idx = y * self.width + x
        if not (0 <= x < self.width) or not (0 <= y < self.height) or self.board[idx] != 0:
            print("Illegal move:", x, y, file=sys.stderr)
            return False
        self._make_move(idx)
        return True

    def score(self, args):
        p1, p2 = self._compute_scores_fast()
        print(p1, p2)
        return True

    # ---------- Move application / undo ----------

    def _make_move(self, idx):
        self.current_hash ^= self.zobrist[idx][self.to_play]
        self.current_hash ^= self.zobrist_turn
        self.board[idx] = self.to_play
        self.empty_set.discard(idx)
        self.to_play = 3 - self.to_play

    def _undo_move(self, idx):
        self.to_play = 3 - self.to_play
        self.current_hash ^= self.zobrist_turn
        self.current_hash ^= self.zobrist[idx][self.to_play]
        self.board[idx] = 0
        self.empty_set.add(idx)

    # ---------- Scoring (true PoE2 score, reference implementation) ----------

    def _compute_scores_fast(self):
        """Exact PoE2 scoring using the reference algorithm, with caching."""
        h = self.current_hash
        cached = self.score_cache.get(h)
        if cached is not None:
            return cached

        w, ht = self.width, self.height
        board = self.board

        p1_score = 0
        p2_score = 0  # handicap added at the end

        # Progress from left-to-right, top-to-bottom
        for y in range(ht):
            for x in range(w):
                idx = y * w + x
                c = board[idx]
                if c == 0:
                    continue

                lone_piece = True  # assume lone until proven otherwise

                # ----- Horizontal -----
                hl = 1
                if x == 0 or board[y * w + (x - 1)] != c:
                    x1 = x + 1
                    while x1 < w and board[y * w + x1] == c:
                        hl += 1
                        x1 += 1
                else:
                    lone_piece = False

                # ----- Vertical -----
                vl = 1
                if y == 0 or board[(y - 1) * w + x] != c:
                    y1 = y + 1
                    while y1 < ht and board[y1 * w + x] == c:
                        vl += 1
                        y1 += 1
                else:
                    lone_piece = False

                # ----- Diagonal (down-right) -----
                dl = 1
                if y == 0 or x == 0 or board[(y - 1) * w + (x - 1)] != c:
                    x1 = x + 1
                    y1 = y + 1
                    while x1 < w and y1 < ht and board[y1 * w + x1] == c:
                        dl += 1
                        x1 += 1
                        y1 += 1
                else:
                    lone_piece = False

                # ----- Anti-diagonal (down-left) -----
                al = 1
                if y == 0 or x == w - 1 or board[(y - 1) * w + (x + 1)] != c:
                    x1 = x - 1
                    y1 = y + 1
                    while x1 >= 0 and y1 < ht and board[y1 * w + x1] == c:
                        al += 1
                        x1 -= 1
                        y1 += 1
                else:
                    lone_piece = False

                # ----- Add scores for found lines -----
                for line_length in (hl, vl, dl, al):
                    if line_length > 1:
                        add = POW2[line_length - 1]  # 2^(L-1)
                        if c == 1:
                            p1_score += add
                        else:
                            p2_score += add

                # ----- Lone piece special case -----
                if hl == vl == dl == al == 1 and lone_piece:
                    if c == 1:
                        p1_score += 1
                    else:
                        p2_score += 1

        result = (p1_score, p2_score + self.handicap)
        self.score_cache[h] = result
        return result

    # ---------- Aggressive evaluation ----------

    def _evaluate_position(self):
        """Aggressive evaluation: exact score + potential threats."""
        p1, p2 = self._compute_scores_fast()

        me = self.to_play
        opp = 3 - me
        my_score = p1 if me == 1 else p2
        opp_score = p2 if me == 1 else p1

        bonus = 0
        board = self.board

        for idx in self.empty_set:
            x, y = idx % self.width, idx // self.width
            lines = self.line_cells[idx]

            my_potential = 0
            opp_potential = 0

            for neg_cells, pos_cells in lines:
                # contiguous my stones from this empty cell
                my_neg = 0
                for c in neg_cells:
                    if board[c] == me:
                        my_neg += 1
                    else:
                        break
                my_pos = 0
                for c in pos_cells:
                    if board[c] == me:
                        my_pos += 1
                    else:
                        break
                total_my = my_neg + my_pos

                # contiguous opp stones from this empty cell
                opp_neg = 0
                for c in neg_cells:
                    if board[c] == opp:
                        opp_neg += 1
                    else:
                        break
                opp_pos = 0
                for c in pos_cells:
                    if board[c] == opp:
                        opp_pos += 1
                    else:
                        break
                total_opp = opp_neg + opp_pos

                # Aggressive: heavily reward our own larger chains
                if total_my >= 1:
                    my_potential += POW2[total_my] * 4

                # Still respect blocking, but slightly less
                if total_opp >= 1:
                    opp_potential += POW2[total_opp] * 3

            bonus += my_potential - opp_potential

        # Aggressive: weight our own score slightly more than opponent’s
        raw = (my_score * 12 - opp_score * 10) + bonus
        return raw

    # ---------- Frontier moves (locality) ----------

    def _get_frontier_moves(self):
        """Return moves near existing stones; fallback to all empties if too small."""
        w, h = self.width, self.height
        board = self.board

        occupied = [i for i in range(w * h) if board[i] != 0]
        if not occupied:
            return list(self.empty_set)  # opening: everything is fine

        frontier = set()
        for idx in occupied:
            x, y = idx % w, idx // w
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        nidx = ny * w + nx
                        if board[nidx] == 0:
                            frontier.add(nidx)

        frontier = frontier & self.empty_set
        if len(frontier) <= 4:
            return list(self.empty_set)
        return list(frontier)

    # ---------- Heuristic move eval for ordering ----------

    def _eval_move(self, idx, player, board):
        """Evaluate move quality for ordering (tactical + blocking + fork bonus)."""
        opp = 3 - player
        score = 0
        lines = self.line_cells[idx]

        directions_with_my_stones = 0

        for neg_cells, pos_cells in lines:
            # extend own lines
            my_neg = 0
            for c in neg_cells:
                if board[c] == player:
                    my_neg += 1
                else:
                    break

            my_pos = 0
            for c in pos_cells:
                if board[c] == player:
                    my_pos += 1
                else:
                    break

            my_total = my_neg + my_pos
            if my_total > 0:
                directions_with_my_stones += 1
                score += POW2[my_total + 1] * 6

            # blocking opponent
            if my_neg == 0:
                opp_neg = 0
                for c in neg_cells:
                    v = board[c]
                    if v == opp:
                        opp_neg += 1
                    else:
                        break
                if opp_neg > 0 and my_pos == 0:
                    opp_pos = 0
                    for c in pos_cells:
                        v = board[c]
                        if v == opp:
                            opp_pos += 1
                        else:
                            break
                    opp_total = opp_neg + opp_pos
                    if opp_total > 0:
                        score += POW2[opp_total] * 5
            elif my_pos == 0:
                opp_pos = 0
                for c in pos_cells:
                    v = board[c]
                    if v == opp:
                        opp_pos += 1
                    else:
                        break
                if opp_pos > 0:
                    score += POW2[opp_pos] * 5

        # Bonus for forks (multiple directions where we extend our line)
        if directions_with_my_stones >= 2:
            score += 500

        return score

    def _get_ordered_moves(self, tt_move, killer_set, pv_move, player, board, history):
        """Get moves with PV, TT, killers, heuristic, one-ply gain, and center bias."""
        moves = []
        candidates = self._get_frontier_moves()

        for idx in candidates:
            if idx == pv_move:
                priority = 3_000_000_000
            elif idx == tt_move:
                priority = 2_000_000_000
            elif idx in killer_set:
                priority = 1_000_000_000 + history.get(idx, 0)
            else:
                tactical = self._eval_move(idx, player, board)

                # One-ply score gain heuristic
                self._make_move(idx)
                p1, p2 = self._compute_scores_fast()
                self._undo_move(idx)
                if player == 1:
                    gain = p1 - p2
                else:
                    gain = p2 - p1

                hist = history.get(idx, 0)
                priority = tactical * 80 + gain * 40 + hist + self.center_score[idx]

            moves.append((priority, idx))

        moves.sort(reverse=True)
        return [m[1] for m in moves]

    # ---------- Core search: Negamax + AB + TT + PV + heuristics + quiescence ----------

    def _negamax(self, alpha, beta, depth, max_depth, tt, killers, history,
                 start_time, time_budget, node_count, pv_hint, allow_quiescence):
        """Negamax with alpha-beta, TT, killers, history, LMR, PV ordering, quiescence."""
        node_count[0] += 1

        # Time check every ~2k nodes
        if (node_count[0] & 2047) == 0:
            if time.time() - start_time > time_budget:
                raise TimeoutError()

        # Terminal: no moves
        if not self.empty_set:
            p1, p2 = self._compute_scores_fast()
            rel = (p1 - p2) if self.to_play == 1 else (p2 - p1)
            return rel, None, []

        # Leaf or cutoff checks that need true scores
        need_scores = (depth >= max_depth) or (self.score_cutoff < float("inf"))
        if need_scores:
            p1, p2 = self._compute_scores_fast()
            # relative from current player perspective
            rel_score = (p1 - p2) if self.to_play == 1 else (p2 - p1)

            # Score cutoff terminal
            if self.score_cutoff < float("inf"):
                if p1 >= self.score_cutoff or p2 >= self.score_cutoff:
                    return rel_score, None, []

            # Depth limit leaf -> quiescence or aggressive eval
            if depth >= max_depth:
                # Simple quiescence extension: look for big tactical moves
                if allow_quiescence:
                    base_rel = rel_score
                    tactical_moves = []
                    T = 8  # score jump threshold, tunable

                    me = self.to_play

                    for idx in self.empty_set:
                        self._make_move(idx)
                        c1, c2 = self._compute_scores_fast()
                        # After _make_move, to_play has flipped to opponent
                        if me == 1:
                            rel_after = c1 - c2
                        else:
                            rel_after = c2 - c1
                        self._undo_move(idx)

                        if rel_after - base_rel >= T:
                            tactical_moves.append(idx)

                    if tactical_moves:
                        best = -999999999
                        for m in tactical_moves:
                            self._make_move(m)
                            val, _, _ = self._negamax(
                                -beta, -alpha,
                                depth + 1, max_depth + 1,  # small extension
                                tt, killers, history,
                                start_time, time_budget, node_count,
                                None,  # no PV hint inside quiescence
                                False  # no further quiescence extensions
                            )
                            val = -val
                            self._undo_move(m)
                            if val > best:
                                best = val
                        return best, None, []

                eval_val = self._evaluate_position()
                return eval_val, None, []

        h = self.current_hash

        # --- Symmetry reduction hook (optional, not implemented) ---
        tt_key = h

        tt_move = None
        tt_entry = tt.get(tt_key)
        if tt_entry:
            tt_d, tt_flag, tt_val, tt_mv = tt_entry
            if tt_d >= max_depth - depth:
                if tt_flag == EXACT:
                    return tt_val, tt_mv, [tt_mv] if tt_mv is not None else []
                elif tt_flag == LOWER:
                    alpha = max(alpha, tt_val)
                elif tt_flag == UPPER:
                    beta = min(beta, tt_val)
                if alpha >= beta:
                    return tt_val, tt_mv, [tt_mv] if tt_mv is not None else []
            tt_move = tt_mv

        # PV move hint for this depth
        pv_move = None
        if pv_hint is not None and depth < len(pv_hint):
            pv_move = pv_hint[depth]

        killers_at_depth = killers[depth] if depth < len(killers) else [None, None]
        killer_set = set(m for m in killers_at_depth if m is not None)

        moves = self._get_ordered_moves(tt_move, killer_set, pv_move,
                                        self.to_play, self.board, history)

        if not moves:
            # No legal moves -> evaluate position
            p1, p2 = self._compute_scores_fast()
            rel = (p1 - p2) if self.to_play == 1 else (p2 - p1)
            return rel, None, []

        best_value = -999999999
        best_move = moves[0]
        best_pv = []
        tt_flag = UPPER

        for i, move in enumerate(moves):
            self._make_move(move)

            # Late Move Reduction (LMR)
            reduction = 0
            if i >= 3 and depth >= 2 and max_depth - depth >= 2:
                reduction = 1
                if i >= 6:
                    reduction = 2

            child_pv = []
            if i == 0:
                # Full-window search for first (PV) move
                value, _, child_pv = self._negamax(
                    -beta, -alpha,
                    depth + 1, max_depth,
                    tt, killers, history,
                    start_time, time_budget, node_count,
                    pv_hint,
                    allow_quiescence
                )
                value = -value
                child_line = [move] + child_pv
            else:
                # Null window with reduction for non-PV moves
                value, _, _ = self._negamax(
                    -alpha - 1, -alpha,
                    depth + 1, max_depth - reduction,
                    tt, killers, history,
                    start_time, time_budget, node_count,
                    pv_hint,
                    allow_quiescence
                )
                value = -value
                child_line = [move]

                # If we got something interesting, re-search at full window
                if value > alpha and (value < beta or reduction > 0):
                    value, _, child_pv = self._negamax(
                        -beta, -alpha,
                        depth + 1, max_depth,
                        tt, killers, history,
                        start_time, time_budget, node_count,
                        pv_hint,
                        allow_quiescence
                    )
                    value = -value
                    child_line = [move] + child_pv

            self._undo_move(move)

            if value > best_value:
                best_value = value
                best_move = move
                best_pv = child_line

            if value > alpha:
                alpha = value
                tt_flag = EXACT

            if alpha >= beta:
                # Beta cutoff → killer + history updates
                if depth < len(killers):
                    if killers[depth][0] != move:
                        killers[depth][1] = killers[depth][0]
                        killers[depth][0] = move
                history[move] = history.get(move, 0) + (max_depth - depth) ** 2
                tt_flag = LOWER
                break

        tt[tt_key] = (max_depth - depth, tt_flag, best_value, best_move)
        return best_value, best_move, best_pv

    # ---------- genmove: iterative deepening + aspiration window ----------

    def genmove(self, args):
        start_time = time.time()
        # Use ~92% of timelimit as budget
        time_budget = self.time_limit * 0.92

        if not self.empty_set:
            print("0 0")
            return True

        moves = list(self.empty_set)
        if len(moves) == 1:
            idx = moves[0]
            x, y = idx % self.width, idx // self.width
            self._make_move(idx)
            print(x, y)
            return True

        # Save root state
        saved_board = self.board[:]
        saved_to_play = self.to_play
        saved_empty = self.empty_set.copy()
        saved_hash = self.current_hash

        tt = self.tt        # persistent TT across moves
        killers = [[None, None] for _ in range(64)]
        history = {}
        node_count = [0]
        self.score_cache = {}

        best_move = moves[0]
        best_value = -999999999
        pv_hint = None  # principal variation from last completed depth

        max_depth = 1
        try:
            while True:
                # Restore root
                self.board = saved_board[:]
                self.to_play = saved_to_play
                self.empty_set = saved_empty.copy()
                self.current_hash = saved_hash
                self.score_cache.clear()

                if max_depth <= 2:
                    value, move, pv_line = self._negamax(
                        -999999999, 999999999,
                        0, max_depth,
                        tt, killers, history,
                        start_time, time_budget,
                        node_count, pv_hint,
                        True  # allow quiescence at leaves
                    )
                else:
                    # Aspiration window around last best value
                    window = 25
                    alpha = best_value - window
                    beta = best_value + window

                    value, move, pv_line = self._negamax(
                        alpha, beta,
                        0, max_depth,
                        tt, killers, history,
                        start_time, time_budget,
                        node_count, pv_hint,
                        True
                    )

                    if value <= alpha:
                        # Fail low -> widen to true lower bound
                        self.board = saved_board[:]
                        self.to_play = saved_to_play
                        self.empty_set = saved_empty.copy()
                        self.current_hash = saved_hash
                        self.score_cache.clear()
                        value, move, pv_line = self._negamax(
                            -999999999, beta,
                            0, max_depth,
                            tt, killers, history,
                            start_time, time_budget,
                            node_count, pv_hint,
                            True
                        )
                    elif value >= beta:
                        # Fail high -> widen to true upper bound
                        self.board = saved_board[:]
                        self.to_play = saved_to_play
                        self.empty_set = saved_empty.copy()
                        self.current_hash = saved_hash
                        self.score_cache.clear()
                        value, move, pv_line = self._negamax(
                            alpha, 999999999,
                            0, max_depth,
                            tt, killers, history,
                            start_time, time_budget,
                            node_count, pv_hint,
                            True
                        )

                if move is not None:
                    best_move = move
                    best_value = value
                    pv_hint = pv_line

                max_depth += 1

                elapsed = time.time() - start_time
                # Use ~90% of our budget for deepening
                if elapsed > time_budget * 0.9:
                    break

        except TimeoutError:
            # Use best from last completed depth
            pass

        # Restore and play best move
        self.board = saved_board
        self.to_play = saved_to_play
        self.empty_set = saved_empty
        self.current_hash = saved_hash

        x, y = best_move % self.width, best_move // self.width
        self._make_move(best_move)
        print(x, y)
        return True


if __name__ == "__main__":
    interface = CommandInterface()
    interface.main_loop()
