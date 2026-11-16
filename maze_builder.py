import pygame
import random  

alpha = 0.1     # learning rate
gamma = 0.9     # discount factor
epsilon = 0.0   # exploration rate

training = False    
live_eps = 1.0
live_ep = 0
path_cells = []     
show_values = True  
show_numbers = True

episodes=500
max_steps=200

N = 10
CELL = 40
walls = [[0] * N for _ in range(N)]


Q = [[[0.0] * 4 for _ in range(N)] for _ in range(N)]


last_clicked = None     
start = None            
goal = None
start_prev = 0       
goal_prev = 0

pygame.init()
screen = pygame.display.set_mode((N * CELL, N * CELL))
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 16)  

def toggle_special(kind):
    global start, goal, start_prev, goal_prev
    if last_clicked is None:
        return
    r, c = last_clicked

    if kind == "start":
        if start == (r, c):                  
            walls[r][c] = start_prev
            start = None
        elif (r, c) != goal:                
            if start is not None:          
                walls[start[0]][start[1]] = start_prev
            start_prev = walls[r][c]          
            walls[r][c] = 0                
            start = (r, c)
    else:
        if goal == (r, c):
            walls[r][c] = goal_prev
            goal = None
        elif (r, c) != start:
            if goal is not None:
                walls[goal[0]][goal[1]] = goal_prev
            goal_prev = walls[r][c]
            walls[r][c] = 0
            goal = (r, c)

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]   

def step(state, action):
    r, c = state
    dr, dc = ACTIONS[action]
    nr, nc = r + dr, c + dc

    if not (0 <= nr < N and 0 <= nc < N) or walls[nr][nc] == 1:
        return state, -5, False          
    if (nr, nc) == goal:
        return (nr, nc), 100, True      
    return (nr, nc), -1, False         

def choose_action(state, eps):
    if random.random() < eps:
        return random.randrange(4)            

    r, c = state
    values = Q[r][c]
    best = max(values)
    best_actions = [a for a in range(4) if values[a] == best]
    return random.choice(best_actions)        # exploit: best known action

def update(state, action, reward, new_state, done):
    r, c = state
    nr, nc = new_state

    best_next = 0 if done else max(Q[nr][nc])     # value of the best move from where we landed
    old = Q[r][c][action]
    Q[r][c][action] = old + alpha * (reward + gamma * best_next - old)

def reset_q():
    global Q
    Q = [[[0.0] * 4 for _ in range(N)] for _ in range(N)]

def blend(a, b, t):
    """Mix colour a into colour b. t=0 gives a, t=1 gives b."""
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))

def run_episode(eps):
    """One attempt from start until the goal or max_steps. Returns steps taken."""
    state = start
    steps = 0
    for _ in range(max_steps):
        action = choose_action(state, eps)
        new_state, reward, done = step(state, action)
        update(state, action, reward, new_state, done)
        state = new_state
        steps += 1
        if done:
            break
    return steps

def train(episodes, max_steps, eps, verbose=True):
    reset_q()
    steps_log = []

    for ep in range(episodes):
        steps_log.append(run_episode(eps))
        eps = max(0.0 if eps == 0 else 0.05, eps * 0.99)

    if verbose:
        for i in range(0, episodes, 50):
            block = steps_log[i:i + 50]
            print(f"episodes {i:>4}-{i + len(block) - 1:>4}: avg {sum(block) / len(block):6.1f} steps")

def compare(runs):
    for eps0 in (1.0, 0.0):
        lengths = []
        for _ in range(runs):
            train(episodes, max_steps, eps0, verbose=False)   # use eps0, no table
            path = best_path()
            lengths.append(len(path) - 1 if path[-1] == goal else None)
        found = [l for l in lengths if l is not None]
        if found:
            print(f"start_eps={eps0}: solved {len(found)}/{runs}, "
                  f"shortest {min(found)}, average {sum(found) / len(found):.1f}")
        else:
            print(f"start_eps={eps0}: solved 0/{runs}")

def best_path():
    """Follow the highest-scoring action from the start, with no randomness."""
    state, path = start, [start]
    for _ in range(100):
        state, _, done = step(state, choose_action(state, 0))
        path.append(state)
        if done:
            break
    return path

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            row, col = y // CELL, x // CELL
            last_clicked = (row, col)
            training = False          
            path_cells = []          
            reset_q()                 
            if (row, col) != start and (row, col) != goal:
                walls[row][col] = 1 - walls[row][col]

        if event.type == pygame.KEYDOWN:
            if event.mod & pygame.KMOD_SHIFT:
                if event.key == pygame.K_s:
                    toggle_special("start")
                elif event.key == pygame.K_g:
                    toggle_special("goal")
            elif start and goal:
                if event.key == pygame.K_t:
                    train(episodes, max_steps, epsilon)
                    path = best_path()
                    print("Reached goal:", path[-1] == goal, "in", len(path) - 1, "steps")
                    print(path)
                elif event.key == pygame.K_c:
                    compare(20)
                elif event.key == pygame.K_n:
                    show_numbers = not show_numbers
                elif event.key == pygame.K_l:
                    reset_q()
                    training = True
                    live_eps = 1.0
                    live_ep = 0
                    path_cells = []

            if event.key == pygame.K_v:
                show_values = not show_values

    if training and start and goal:
        run_episode(live_eps)
        live_eps = max(0.05, live_eps * 0.99)
        live_ep += 1
        pygame.display.set_caption(f"Episode {live_ep}/{episodes}   epsilon {live_eps:.2f}")
        if live_ep >= episodes:
            training = False
            path_cells = best_path()
    
        # work out the range of values so we can scale colours
    visited = [max(Q[r][c]) for r in range(N) for c in range(N)
               if walls[r][c] == 0 and any(Q[r][c])]
    lo, hi = (min(visited), max(visited)) if visited else (0.0, 1.0)
    if hi == lo:
        hi = lo + 1

    screen.fill((200, 200, 200))
    for r in range(N):
        for c in range(N):
            if (r, c) == start:
                color = (27, 138, 122)
            elif (r, c) == goal:
                color = (217, 138, 28)
            elif walls[r][c]:
                color = (30, 40, 50)
            elif show_values and any(Q[r][c]):
                t = (max(Q[r][c]) - lo) / (hi - lo)       # 0 = worst seen, 1 = best seen
                color = blend((250, 250, 245), (70, 130, 210), t)
            else:
                color = (250, 250, 245)
            pygame.draw.rect(screen, color, (c * CELL + 1, r * CELL + 1, CELL - 2, CELL - 2))

            if show_numbers and walls[r][c] == 0 and (r, c) != goal and any(Q[r][c]):
                v = max(Q[r][c])
                label = f"{v:.0f}" if abs(v) >= 10 else f"{v:.1f}"
                dark_tile = sum(color) / 3 < 140          # pick a readable text colour
                img = font.render(label, True, (255, 255, 255) if dark_tile else (30, 40, 50))
                screen.blit(img, img.get_rect(center=(c * CELL + CELL // 2, r * CELL + CELL // 2)))

    for (r, c) in path_cells:                              # learned route
        pygame.draw.circle(screen, (220, 60, 60), (c * CELL + CELL // 2, r * CELL + CELL // 2), 5)

    if last_clicked:
        r, c = last_clicked
        pygame.draw.rect(screen, (220, 60, 60), (c * CELL, r * CELL, CELL, CELL), 3)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()