directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

def num_to_cord(action):
    return((int(action/8),action % 8))
def cord_to_num(cord):
    return(cord[0]*8+cord[1])  

def positionLegal(cord):
    if (cord[0]<0 or cord[0]>=8 or cord[1]<0 or cord[1]>=8):
        return False
    else:
        return True
    
def place(board, enables, player):
    max_reverse_num = 0
    max_reverse_action = 65
    for i in range(len(enables)):
        reverse_num = countReverse(board, enables[i], player)
        if reverse_num > max_reverse_num:
            max_reverse_num = reverse_num
            max_reverse_action = enables[i]
    return (max_reverse_action)

def countReverse(board, action, player):  # 这里就是 进行下棋操作。并更新棋盘
    cord = num_to_cord(action)
    reverse_sum = 0
    for direction in directions:
        count = 0
        current_position = (cord[0]+direction[0],cord[1]+direction[1])
        while positionLegal(current_position):
            if board[(player+1) % 2][current_position[0]][current_position[1]]:                
                count += 1
                current_position = (current_position[0]+direction[0],current_position[1]+direction[1])
            else:
                break
        if (positionLegal(current_position)) and (board[player][current_position[0]][current_position[1]]):
            reverse_sum = reverse_sum + count
    return reverse_sum
                
