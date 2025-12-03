import subprocess
import sys
import os
import select
import math

p1_name = sys.argv[1]
p2_name = sys.argv[2]

if not os.path.isfile(p1_name):
    print("Could not find file", p1_name)
if not os.path.isfile(p2_name):
    print("Could not find file", p2_name)

p1_proc = subprocess.Popen(
    ["python3", p1_name],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)
p2_proc = subprocess.Popen(
    ["python3", p2_name],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

xsize = 7
ysize = 7
handicap = 5.5
score_cutoff = 0   # 0 means "no cutoff"
timelimit = 1


def send_command(process, name, command):
    print(name, ":", command)
    process.stdin.write(command + "\n")
    process.stdin.flush()
    output = ""
    line = process.stdout.readline()
    try:
        while line and line[0] != "=":
            if len(line.strip()) > 0:
                output += line
            line = process.stdout.readline()
            # Non-blocking stderr read
            readable, _, _ = select.select([process.stderr], [], [], 0)
            if readable:
                err = process.stderr.read()
                if err:
                    sys.stderr.write(err)
                    sys.stderr.flush()
    except Exception:
        sys.exit()
    print(output, end="")  # avoid double newline
    return output


def play_game(p1, p1_name, p2, p2_name):
    print("P1:", p1_name)
    print("P2:", p2_name)

    params = " ".join([str(p) for p in [xsize, ysize, handicap, score_cutoff]])
    send_command(p1, p1_name, "init_game " + params)
    send_command(p2, p2_name, "init_game " + params)
    send_command(p1, p1_name, "timelimit " + str(timelimit))
    send_command(p2, p2_name, "timelimit " + str(timelimit))

    # Real cutoff value (inf if score_cutoff == 0)
    if score_cutoff == 0:
        real_score_cutoff = math.inf
    else:
        real_score_cutoff = score_cutoff

    player = p1
    p_name = p1_name
    opp = p2
    o_name = p2_name

    p1_score = 0.0
    p2_score = 0.0

    for _ in range(xsize * ysize):
        # Ask current player for move
        move = send_command(player, p_name, "genmove").strip()
        # Relay move to opponent
        send_command(opp, o_name, "play " + move)
        # Show and score from current player's perspective
        send_command(player, p_name, "show")
        score = send_command(player, p_name, "score").strip()
        # Parse scores "p1 p2"
        parts = score.split()
        if len(parts) >= 2:
            p1_score, p2_score = [float(x) for x in parts[:2]]
        else:
            # If something weird happens, break
            break

        # Cutoff checks (return numeric difference!)
        if p1_score > real_score_cutoff:
            print(p1_name, "wins by score cutoff.")
            return p1_score - p2_score
        if p2_score > real_score_cutoff:
            print(p2_name, "wins by score cutoff.")
            return p1_score - p2_score

        # Swap players
        tmp = player
        tmp_name = p_name
        player = opp
        p_name = o_name
        opp = tmp
        o_name = tmp_name

    # Full board or loop done: decide winner by points
    if p1_score > p2_score:
        print(p1_name, "wins by", p1_score - p2_score, "points.")
    else:
        print(p2_name, "wins by", p2_score - p1_score, "points.")

    return p1_score - p2_score


# First game: p1 is P1, p2 is P2
p1_score = play_game(p1_proc, p1_name, p2_proc, p2_name)
# Second game: swap sides; note we subtract because now p2 is "P1"
p1_score += -play_game(p2_proc, p2_name, p1_proc, p1_name)

print("Overall winner:")
if p1_score > 0:
    print(p1_name, "by a", p1_score, "point difference.")
else:
    print(p2_name, "by a", -p1_score, "point difference.")
